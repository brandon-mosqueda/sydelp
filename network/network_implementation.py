from __future__ import annotations

import time
import pykka
from nodes.node import Node

NODES_TYPES = ['node', 'user_interface', 'data_logger']

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
    node: Node | None # actual node object to be associated with the actor, we can use all the functions in the class and define the node in different ways
    name: str
    ID: int
    partners: list | None # node neighbors
    running: bool
    type: str
    round: int
    messsages_from_trainers: list | None # messages from the trainers/attackers at each round
    num_participants: list | None # number of participants in the training round (trainers) at each round

    def __init__(self, name, type, node, ID):
        super().__init__()
        self.name = name
        self.partners: list = None  # Partner actor reference
        self.running = True
        if type not in NODES_TYPES:
            raise ValueError("Invalid type")
        self.type = type
        self.round = 0

        # the node is an object of the class Node if the NodeActor is not a data_logger or user_interface
        # the node is None if the NodeActor is a data_logger or user_interface, it will not participate in the training
        if type != 'data_logger' and type != 'user_interface':
            self.ID = node.ID
            self.node = node

        if type == 'data_logger':
            self.messsages_from_trainers = []
            self.num_participants = []

    def record_messages(self, message):
        print(f"{self.name} received message: {message}")

        # record the message
        if message['command'] == 'new_model':
            round = message['round']
            if self.messsages_from_trainers[round] is None:
                self.messsages_from_trainers[round] = 0
                self.num_participants[round] = self.node.num_participants

            self.messsages_from_trainers[round] += 1


    def set_partner(self, partners):
        """Set the partner actor."""
        self.partners = partners

    def read_new_block(self, block):
        """Read the new block from the blockchain."""


        # corner case there are two or more verifiers, so check if the block is already processed by checking the round
        if block['round'] == self.round:
            return

        # check the score next to the node ID
        new_score = block['scores'][self.ID]

        # update the score of the node
        # TODO

        # do the new training
        # TODO: with the built-in node object

        # calculate the proof of work
        # TODO

        # generate the signature
        # TODO

        # broadcast our new model


        # TODO: this is just the structure of the message, we need to implement the actual functions, like signature generation, proof of work, etc.
        msg = {
            "command": "new_model",
            "model": self.node.get_model_weights(),
            "signature": "signature",
            "proof_of_work": "proof_of_work",
            "public_key": "public_key",
            "ID": self.ID,
            "round": self.round
        }

        self.broadcast(msg)

        # update the round
        self.round += 1

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
        if self.type == 'trainer':
            if message['command'] == 'new_block':
                self.read_new_block(message['block'])
            return

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

    def start_interface(self):
        print(f"{self.name} is starting.")
        print(f"{self.name} is ready to receive commands.")

        list_of_commands = ["1- start process", "2- stop all nodes", "3 + ID- stop a specific node", "4- stop the process", "5 + 'trainer'|'attacker'- add a new node"]

        while 1:
            print("List of commands:")
            for command in list_of_commands:
                print(command)

            command = input("Enter command: ")

            if command == "1":
                self.broadcast("start")
            elif command == "2":
                self.broadcast("stop")
            elif command == "4":
                self.broadcast("stop")
                break
            elif command[0] == "3":
                # TODO check typos and catch exceptions
                self.partners[int(command[2])].tell("stop")
            elif command[0] == "5":
                # the idea is, I add a new node, then broadcast the new list of partners to everyone
                # the verifier the will add the new partners and send them a message to start the training
                # the verifier will give the maximum score to the new node
                #TODO: implement the verifier part, the other nodes doesn't even need to know the type of the new node
                if command[2:] == "trainer":
                    new_node = NodeActor.start(f"Node_{len(self.partners)}", 'trainer', Node(), len(self.partners))
                    self.partners.append(new_node)
                elif command[2:] == "attacker":
                    new_node = NodeActor.start(f"Node_{len(self.partners)}", 'attacker', Node(), len(self.partners))
                    self.partners.append(new_node)

                self.broadcast("update_partners", self.partners)
            else:
                print("Invalid command")

            print("Command executed\n\n")

        self.stop()


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

        # define 1 verifier for simplicity sake, in the future we can define more verifiers
        elif i == 2:
            node = NodeActor.start(f"Node_{i}", 'verifier')

        else:
            # here we will defined different types of nodes based on the protocol (i think we'll implement only sydelp)
            # for now we just implement the sydel protocol
            if params['protocol'] == 'sydelp':
                # TODO: define the node inside the actor
                node = NodeActor.start(f"Node_{i}", 'trainer', Node(), i)

        nodes.append(node)


    for i in range(MAX_NODES):
        list_of_partners = [node for node in nodes if node != nodes[i]]
        nodes[i].tell({"command": "set_partner", "partner": list_of_partners})

    nodes[0].proxy().start_interface()

    # TODO: we should send a message to start the first round of training

    # nodes[1].proxy().run()
    nodes[0].proxy().command_input()
