import json
import threading
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from pandas import DataFrame

from actors_nodes.node_actor import NodeActor  # Assuming NodeActor is defined elsewhere
from nodes.node import Node
from utils.typing import Float


class DataLoggerActor(NodeActor):
    """
    Data Logger Actor class

    This class is used to implement the data logger actor in the federated learning process.

    The idea is that this class is passive and only records the data from the trainers and the user interface.

    At the end of the process, the data logger can be queried to get the statistics of the process.

    The plot part is not already finished, but the idea is to have a dynamic plot that shows the number of messages
    """

    messsages_from_trainers: list  # List of messages received per round
    num_participants: list  # List of participants per round
    num_malicious_participants: list  # List of malicious participants per round
    quality: dict  # Dictionary of the quality of the model per round
    util_node: Node  # Reference to the utility node used for simplicity in the quality calculation
    X_test: DataFrame
    y_test: DataFrame

    def __init__(self, node_id: int, util_node, X_test, y_test, metrics_params, init_num_attacker, honest_nodes, simulation_name) -> None:
        super().__init__(node_id)
        self.messsages_from_trainers = []
        self.num_participants = honest_nodes + init_num_attacker
        self.honest_participants = honest_nodes
        self.attacker_per_round = []
        self.current_num_attacker = init_num_attacker
        self.quality = {}
        self.util_node = util_node
        self.X_test = X_test
        self.y_test = y_test
        self.metrics_params = metrics_params
        self.simulation_name = simulation_name
        self.scores_per_round = {}
        print("X_test: ", X_test)
        print("y_test: ", y_test)


        # # Initialize Matplotlib figure
        # self.fig, (self.ax1, self.ax3) = plt.subplots(2, 1, figsize=(10, 8))
        # self.ax2 = self.ax1.twinx()  # Secondary y-axis for the first plot
        #
        # # First subplot: Messages and participants
        # self.ax1.set_xlabel("Round")
        # self.ax1.set_ylabel("Messages", color="blue")
        # self.ax2.set_ylabel("Participants", color="orange")
        # self.ax1.set_title("Messages and Participants per Round")
        # self.line1, = self.ax1.plot([], [], 'b-', label="Messages")  # Messages line
        # self.line2, = self.ax2.plot([], [], 'orange', label="Participants")  # Participants line
        # self.line3, = self.ax2.plot([], [], 'r-', label="Malicious Participants")  # Malicious line
        #
        # # Second subplot: Quality of the model
        # self.ax3.set_xlabel("Round")
        # self.ax3.set_ylabel("Quality", color="green")
        # self.ax3.set_title("Model Quality per Round")
        # self.line4, = self.ax3.plot([], [], 'g-', label="Model Quality")
        #
        # # Start dynamic animation (non-blocking)
        # self.start_animation()
        #
        # # Start the background thread to update the plot dynamically
        # threading.Thread(target=self.background_plot_updates, daemon=True).start()

    def on_receive(self, message: dict):
        """Handle incoming messages."""
        # Handle messages from trainers
        if message.get('command') == 'new_model' or message.get('command') == 'new_block':
            self.record_messages(message)
            return

        if message.get('command') == 'new_attacker':
            self.current_num_attacker += 1
            return

        # Handle messages from the user interface
        if message.get('command') == 'print_messages_stats':
            print(f"Messages from trainers: {self.messsages_from_trainers}")
            print(f"Number of participants: {self.num_participants}")
            print(f"Quality of the model: {self.quality}")
            return

        if message.get('command') == 'scores':
            self.scores_per_round[message['round']] = {
                "attackers_scores": message['attackers_scores'],
                "honest_scores": message['honest_scores']
            }
            return

        # Handle primitive messages [stop, hello, set_partner, update_partners]
        super().on_receive(message)

    def record_messages(self, message):
        """Record the messages and update data."""

        if message.get('command') == 'new_block':
            # TODO: revise this for coherence with the other functions, not [] but .get()
            self.util_node.set_model_weights(message['model'])
            self.quality[message['round']] = self.calculate_quality()
            self.attacker_per_round.append(self.current_num_attacker)


    def update_plot(self, frame=None):
        """Update the plot dynamically."""
        rounds = list(range(1, len(self.messsages_from_trainers) + 1))

        # Update the first subplot
        self.line1.set_data(rounds, self.messsages_from_trainers)
        self.line2.set_data(rounds, self.num_participants)
        self.line3.set_data(rounds, self.num_malicious_participants)

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        # Update the second subplot
        self.line4.set_data(rounds, self.quality)

        self.ax3.relim()
        self.ax3.autoscale_view()

        return self.line1, self.line2, self.line3, self.line4

    def start_animation(self):
        """Start the dynamic visualization."""
        plt.ion()  # Enable interactive mode
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=1000, cache_frame_data=False)
        self.fig.legend(loc="upper center", ncol=4)
        plt.tight_layout()
        plt.draw()  # Draw the plot without blocking

    def background_plot_updates(self):
        """Background thread to keep the plot responsive."""
        while True:
            plt.pause(0.01)  # Allows the GUI event loop to update the plot
            time.sleep(0.01)  # Prevents excessive CPU usage

    def write_to_file(self):
        # open "./results/result_network.json" in read mode
        print("Writing to file")
        try:
            with open("./results/result_network.json", "r") as file:
                # read the content of the file
                data = file.read()
                # convert the content to a dictionary
                data = eval(data)
        except FileNotFoundError:
            data = {}
            print("File not found, creating a new one.")
        print(data)

        # add the data in this manner
        """
        {
            self.simulation_name: {
                "quality": self.quality,
                "attacker_per_round": self.attacker_per_round
                "honest_participants": self.honest_participants
            }
        """
        data[self.simulation_name] = {
            "quality": self.quality,
            "attacker_per_round": self.attacker_per_round,
            "honest_participants": self.honest_participants,
            "scores_per_round": self.scores_per_round
        }

        print(data)

        try:
            # open "./results/result_network.json" in write mode
            with open("./results/result_network.json", "w") as file:
                # write the dictionary to the file
                json.dump(data, file, indent=4)
        except FileNotFoundError:
            print("Failed to write the data to the file.")

    def on_stop(self):
        """Stop the actor."""
        # print the quality of the model
        try:
            print(f"\n\nQuality of the model: {self.quality}\n\n")
        except IndexError:
            print("No model quality recorded.")

        self.write_to_file()

        super().on_stop()

    def round_prediction(self) -> DataFrame:
        # This function is the adaptation of the round_prediction function in the Learning class
        preds: DataFrame = DataFrame(self.util_node.model.predict(
            self.X_test,
            verbose=0
        ))

        # For binary classification
        if preds.shape[1] == 1:
            print("checkpoint 1.1 binary classification")
            preds = pd.concat([1 - preds[0], preds], axis=1)
            preds.columns = [0, 1]


        preds['predicted'] = np.argmax(preds, axis=1)
        preds['observed'] = self.y_test



        return preds

    def calculate_quality(self):
        # This function is the adaptation of the round_metrics function in the Learning class
        predictions = self.round_prediction()
        # print("predictions: ", predictions)

        try:
            loss = self.util_node.model.evaluate(
                self.X_test,
                self.y_test,
                verbose=0
            )
        except Exception as e:
            print(f"Error in the evaluation of the model: {e}")
            return
        # print("loss: ", loss)

        try:
            round_metrics: dict[str, Float] = {
                metric: self.metrics_params[metric]['function'](
                    y_true=predictions['observed'].to_numpy().astype("int"),
                    y_pred=predictions['predicted'].to_numpy().astype("int"),
                    **self.metrics_params[metric]['params']
                ) for metric in self.metrics_params
            }
        except Exception as e:
            print(f"Error in the calculation of the metrics: {e}")
            return

        round_metrics['loss'] = loss

        print("round_metrics: ", round_metrics)

        return round_metrics['accuracy']
