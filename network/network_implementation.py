from __future__ import annotations

import time
import pykka
from nodes.node import Node

NODES_TYPES = ['trainer', 'attacker', 'verifier', 'user_interface', 'data_logger']

# sydelp protocol global variables
HONEST_TRAINER_SYDELP_NUM = 3
MALICIOUS_TRAINER_SYDELP_NUM = 1
VERIFIER_SYDELP_NUM = 1

# sybilwall protocol global variables
HONEST_TRAINER_SYBILWALL_NUM = 3
MALICIOUS_TRAINER_SYBILWALL_NUM = 1


class NodeActor(pykka.ThreadingActor):
    """Example actor class that sends and receives messages
    to and from a partner actor.
    """
    node: Node | None
    name: str
    partners: list | None # node neighbors
    running: bool
    type: str

    def __init__(self, name, type):
        super().__init__()
        self.name = name
        self.partners: list = None  # Partner actor reference
        self.running = True
        if type not in NODES_TYPES:
            raise ValueError("Invalid type")
        self.type = type

    def record_messages(self, message):
        print(f"{self.name} received message: {message}")

    def set_partner(self, partners):
        """Set the partner actor."""
        self.partners = partners

    def on_receive(self, message):
        """Handle incoming messages."""

        # CORNER CASES MESSAGE HANDLING

        # If the message is a string "stop", stop the actor
        if message == "stop":
            self.running = False
            self.stop()
            return

        # If the message is a dictionary with the key "command" and the value "set_partner",
        if isinstance(message, dict) and message.get('command') == 'set_partner':
            self.set_partner(message['partner'])
            print(f"{self.name} set partner to {message['partner']}")
            return

        # If is the special node 1 just record the message
        if self.type == 'data_logger':
            self.record_messages(message)
            return

        # NORMAL MESSAGE HANDLING
        print(f"{self.name} received message: {message}")

        # Sydelp Protocol

        # 1. receiving the start command to initiate training
        # TODO

        # 2. receiving the message from the training nodes and process it
        # (verifier)
        # TODO

        # 3. receiving the message from the verifier
        # (trainer)
        # TODO

        # SybilWall Protocol

        # 1. receiving the start command to initiate training
        # TODO

        # 2. receiving messages from neighbors
        # (trainer)
        # TODO

        # 3. sending messages to neighbors
        # (trainer)
        # TODO

    def broadcast(self, message):
        """Send a message to the partner actor."""
        if self.partners:
            for partner in self.partners:
                # print(f"{self.name} sending message to {partner}")
                try:
                    partner.tell(message)
                except pykka.ActorDeadError:
                    print(f"{self.name} failed to send message to {partner}")

    def command_input(self):
        command = ""
        while command != "stop":
            time.sleep(1)
            command = input("Enter command: ")
            if command == "stop":
                self.broadcast("stop")
                self.running = False
                time.sleep(1)
                print(f"{self.name} is stopping.")
                self.stop()
            else:
                self.broadcast(command)

    def on_stop(self):
        """Clean up when the actor stops."""
        print(f"{self.name} is stopping.")


MAX_NODES = 100


def network_implementation(params):
    nodes = []

    MAX_NODES = params['nodes_num']

    if params['protocol'] == 'sydelp':
        HONEST_TRAINER_SYDELP_NUM = params['honest_trainer_num']
        MALICIOUS_TRAINER_SYDELP_NUM = params['malicious_trainer_num']
        VERIFIER_SYDELP_NUM = params['verifier_num']

    for i in range(MAX_NODES):
        if i == 0:
            node = NodeActor.start(f"Node_{i}", 'user_interface')
        elif i == 1:
            node = NodeActor.start(f"Node_{i}", 'data_logger')

        # for now we just implement the sydel protocol
        if params['protocol'] == 'sydelp':
            # TODO: associate the other nodes with the other types
            # idea each node actor has a node object associated with it, so we can use the already implemented classes
            if i < HONEST_TRAINER_SYDELP_NUM + 2:
                node = NodeActor.start(f"Node_{i}", 'trainer')
            elif i < HONEST_TRAINER_SYDELP_NUM + MALICIOUS_TRAINER_SYDELP_NUM + 2:
                node = NodeActor.start(f"Node_{i}", 'attacker')
            elif i < HONEST_TRAINER_SYDELP_NUM + MALICIOUS_TRAINER_SYDELP_NUM + VERIFIER_SYDELP_NUM + 2:
                node = NodeActor.start(f"Node_{i}", 'verifier')
            nodes.append(node)

    for i in range(MAX_NODES):
        list_of_partners = [node for node in nodes if node != nodes[i]]
        nodes[i].tell({"command": "set_partner", "partner": list_of_partners})

    # nodes[1].proxy().run()
    nodes[0].proxy().command_input()
