import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from actors_nodes.node_actor import NodeActor  # Assuming NodeActor is defined elsewhere


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
    quality: list  # List of model quality per round

    def __init__(self, node_id: int) -> None:
        super().__init__(node_id)
        self.messsages_from_trainers = []
        self.num_participants = []
        self.num_malicious_participants = []
        self.quality = []

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

        # Handle messages from the user interface
        if message.get('command') == 'print_messages_stats':
            print(f"Messages from trainers: {self.messsages_from_trainers}")
            print(f"Number of participants: {self.num_participants}")
            print(f"Number of malicious participants: {self.num_malicious_participants}")
            print(f"Quality of the model: {self.quality}")
            return

        # Handle primitive messages [stop, hello, set_partner, update_partners]
        super().on_receive(message)

    def calculate_quality(self, model):
        """Mock function to calculate the quality of the model."""
        # TODO: Implement a real function to calculate model quality
        return 1.0

    def record_messages(self, message):
        """Record the messages and update data."""
        # print(f"{self.ID} received message: {message}")

        # Record the message based on command
        # if message.get('command') == 'new_model':
        #     round = message['round']
        #     # Ensure lists have enough space
        #     while len(self.messsages_from_trainers) <= round:
        #         self.messsages_from_trainers.append(None)
        #         self.num_participants.append(None)
        #         self.num_malicious_participants.append(None)
        #         self.quality.append(None)
        #
        #     if self.messsages_from_trainers[round] is None:
        #         self.messsages_from_trainers[round] = 0
        #         self.num_participants[round] = self.node.num_participants
        #         self.num_malicious_participants[round] = self.node.num_malicious_participants
        #
        #     self.messsages_from_trainers[round] += 1

        if message.get('command') == 'new_block':
            # TODO: revise this for coherence with the other functions, not [] but .get()
            if self.quality[message['round']] is None:
                self.quality[message['round']] = self.calculate_quality(message['model'])
        
        # self.update_plot()

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

    def on_stop(self):
        """Stop the actor."""
        # print the quality of the model
        try:
            print(f"\n\nQuality of the model: {self.quality}\n\n")
        except IndexError:
            print("No model quality recorded.")

        # # Stop the plot
        # plt.ioff()
        # plt.show()

        # Stop the actor
        super().on_stop()