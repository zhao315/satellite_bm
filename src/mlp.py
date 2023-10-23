#/usr/bin/env python3
import os
import pickle
import numpy as np
from rich import print

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from ray import train, tune
from ray.train import Checkpoint, RunConfig
from ray.tune.schedulers import ASHAScheduler

from load_data import load_data
from visualize_func import visualize_func


class NeuralNetwork(nn.Module):
    def __init__(self, config, in_dim=9, out_dim=1):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        layer_config = [
            config["l0"], config["l1"], config["l2"], config["l3"],
            config["l4"], config["l5"], config["l6"], config["l7"],
            config["l8"], config["l9"]
        ]
        self.layer_sizes = [in_dim] + layer_config[:config["layer_number"]] + [out_dim]

        self.layer_list = []
        for idx in range(len(self.layer_sizes)-1):
            self.layer_list.append(
                nn.Linear(self.layer_sizes[idx], self.layer_sizes[idx+1])
            )
        self.layers = nn.ModuleList(self.layer_list)

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


def train_func(config):
    """training function"""
    model = NeuralNetwork(config=config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    loss_fn  = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    train_dataloader, val_dataloader, test_dataloader = load_data(
        data_dir=config["data_dir"],
        data_name=config["data_name"],
        split_ratio=config["split_ratio"],
        batch_size=config["batch_size"],
    )

    for epoch in range(config["epochs"]):
        running_loss = 0.0
        epoch_steps = 0

        # training 
        model.train()

        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device, dtype=torch.float32) 
            labels = labels.to(device, dtype=torch.float32) 

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 100 == 99:
                print("[%d, %5d], loss: %.3f" % (epoch+1, i+1, running_loss / labels.size(0)))
                running_loss = 0.0

        # validation
        model.eval()
    
        val_loss = 0.0
        val_steps = 0.0

        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(device, dtype=torch.float32) 
                labels = labels.to(device, dtype=torch.float32) 

                outputs = model(inputs)

                loss = loss_fn(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()),
            "my_model/checkpoint.pt",
        )

        checkpoint = Checkpoint.from_directory("my_model")
        train.report(
            {"loss": val_loss / val_steps},
            checkpoint=checkpoint,
        )
    print("Finish Training")


# test function
def test_best_model(best_result):
    best_trained_model = NeuralNetwork(config=best_result.config)

    print("⚡" * 80)
    print("The structure of the model: \n", best_trained_model)
    print("⚡" * 80)
    print()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_trained_model.to(device)

    checkpoint_path = os.path.join(
        best_result.checkpoint.to_directory(), "checkpoint.pt"
    )

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    train_dataloader, val_dataloader, test_dataloader = load_data(
        best_result.config["data_dir"],
        best_result.config["data_name"],
        best_result.config["split_ratio"],
        best_result.config["batch_size"],

    )
    
    loss_fn = nn.MSELoss()
    total_outputs = []
    total_labels = []

    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            outputs = best_trained_model(inputs)
            loss = loss_fn(outputs,  labels)

            total_outputs.extend(outputs.cpu().numpy().tolist())
            total_labels.extend(labels.cpu().numpy().tolist())


    print("⚡" * 80)
    print(f"Best trail test MSE: {loss:.3f}")
    print("⚡" * 80)
    
    with open("output.pkl", "wb") as fout:
        pickle.dump(total_outputs, fout)

    with open("labels.pkl", "wb") as fout:
        pickle.dump(total_labels, fout)

    with open("loss.pkl", "wb") as fout:
        pickle.dump(loss, fout)

    visualize_func()
        


def main(
    data_dir, data_name,
    num_samples=10, max_num_epochs=10, gpus_per_trial=2
    ):
    config = {
        "l0": tune.sample_from(lambda _: 2 ** np.random.randint(2, 12)),
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 12)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 12)),
        "l3": tune.sample_from(lambda _: 2 ** np.random.randint(2, 12)),
        "l4": tune.sample_from(lambda _: 2 ** np.random.randint(2, 12)),
        "l5": tune.sample_from(lambda _: 2 ** np.random.randint(2, 12)),
        "l6": tune.sample_from(lambda _: 2 ** np.random.randint(2, 12)),
        "l7": tune.sample_from(lambda _: 2 ** np.random.randint(2, 12)),
        "l8": tune.sample_from(lambda _: 2 ** np.random.randint(2, 12)),
        "l9": tune.sample_from(lambda _: 2 ** np.random.randint(2, 12)),
        "layer_number": tune.choice([2, 3, 4, 5, 6, 7, 8, 9]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16, 32]),
        "data_dir": data_dir,
        "data_name": data_name,
        "split_ratio": 0.7,
        "epochs": 20,
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func),
            resources={"cpu": 2, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=RunConfig(
            name="mlp",
            local_dir=os.path.abspath("../result"),
        ),
    )

    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print("⚡" * 50)
    print("Best trial config: {}".format(best_result.config))
    print("Best trail final validation loss: {}".format(best_result.metrics["loss"]))
    print("⚡" * 50)


    print("⚡" * 50)
    print("[bold magenta]test result[/bold magenta]")
    test_best_model(best_result)
    print("⚡" * 50)


if __name__ == "__main__":
    data_dir = os.path.abspath("../notebooks")
    data_name = "p4m_data.csv"

    main(
        data_dir, data_name, num_samples=10, max_num_epochs=10, gpus_per_trial=1
    )
