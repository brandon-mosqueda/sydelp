"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-12-28 12:39:50
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2025-01-24 15:37:51
FilePath: actors_nodes/user_interface_actor.py
Description: 这是默认设置,可以在设置》工具》File Description中进行配置
"""
import time

from actors_nodes.node_actor import NodeActor
from nodes.node import Node


class UserInterfaceActor(NodeActor):
    """
    User Interface Actor class
    
    *Description*
    
    This class is used to implement the user interface actor in the federated learning process.
    
    This actor is responsible for the UI of the process (console based).
    
    This class enables creating new nodes, starting the training process, stopping the training process, and stopping the nodes.
    """
    def __init__(self, node_id: int) -> None:
        super().__init__(node_id)

    def on_receive(self, message: dict):
        if message.get('command') == 'start_interface':
            self.start_interface()
            return

        super().on_receive(message)

    def start_interface(self):
        print(f"{self.ID} is starting.")
        print(f"{self.ID} is ready to receive commands.")

        list_of_commands = ["1 - start process", "2 - stop all nodes", "3 + ID - stop a specific node",
                            "4 - stop the process", "5 + 'trainer'|'attacker' - add a new node",
                            "6 - print the list of active nodes", "7 - print messages log per round"]

        while 1:
            time.sleep(1)
            print("\nList of commands:")
            for command in list_of_commands:
                print(command+"\n")

            command = input("Enter command: ")

            if command == "1":
                self.broadcast({"command": "start"})
            elif command == "2":
                self.broadcast({"command": "stop"})
                self.stop()
                return
            elif command == "4":
                self.broadcast({"command": "stop"})
                self.stop()
                return
            elif command[0] == "3":
                # TODO check typos and catch exceptions
                try:
                    self.partners[int(command[2:])].tell({"command": "stop"})
                except IndexError:
                    print("Invalid index")
            elif command[0] == "5":
                # the idea is, I add a new node, then broadcast the new list of partners to everyone
                # the verifier the will add the new partners and send them a message to start the training
                # the verifier will give the maximum score to the new node
                # TODO: implement the verifier part, the other nodes doesn't even need to know the type of the new node
                # TODO: redo this with the adjustments of the subclasses

                if command[2:] == "trainer":
                    new_node = NodeActor.start(f"Node_{len(self.partners)}", 'trainer', Node(), len(self.partners))
                    self.partners.append(new_node)
                elif command[2:] == "attacker":
                    new_node = NodeActor.start(f"Node_{len(self.partners)}", 'attacker', Node(), len(self.partners))
                    self.partners.append(new_node)

                self.broadcast("update_partners", self.partners)
            elif command == "6":
                print("Number of active nodes: ", len(self.partners) - 2)
            elif command == "7":
                # find data logger and ask for the stats
                self.partners[0].tell({"command": "print_messages_stats"})
            else:
                print("Invalid command")

            print("Command executed\n\n")

        # self.stop()
