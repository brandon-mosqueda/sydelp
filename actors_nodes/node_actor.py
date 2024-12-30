"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-12-28 12:37:50
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-12-30 13:52:55
FilePath: actors_nodes/node_actor.py
Description: 这是默认设置,可以在设置》工具》File Description中进行配置
"""
import pykka


class NodeActor(pykka.ThreadingActor):
    """Example actor class that sends and receives messages
    to and from a partner actor.
    """
    running: bool
    ID: int
    partners: list
    round: int
    

    def __init__(self, ID: int) -> None:
        super().__init__()
        
        print(f"Node {ID} is starting.")
        self.ID = ID
        self.running = True
        self.partners = []
        self.round = 0     
    

    def set_partner(self, partners):
        """Set the partner actor."""
        self.partners = partners
        
    def update_partners(self, partners):
        # recheck this function
        self.partners = partners

    

    def on_receive(self, message):
        """Handle incoming messages."""

        # CORNER CASES MESSAGE HANDLING
        
        # if the message is not a dictionary, print an error message
        if not isinstance(message, dict):
            print(f"{self.ID} received an invalid message: {message}")
            return

        # 1- If the message is a string "stop", stop the actor
        if message.get('command') == 'stop':
            self.stop()
            return

        # 2- If the message is a dictionary with the key "command" and the value "set_partner",
        if message.get('command') == 'set_partner':
            self.set_partner(message['partner'])
            print(f"{self.ID} set partner to {message['partner']}")
            return

        # 3- If the message is a dictionary with the key "command" and the value "update_partners",
        if message.get('command') == 'update_partners':
            self.update_partners(message['partners'])
            print(f"{self.ID} updated partners to {message['partners']}")
            return
        
        # 4- If the message is not recognized, print an error message
        print(f"{self.ID} received an invalid message: {message}")
        
        # NORMAL MESSAGE HANDLING are in the subclasses
        

    def broadcast(self, message):
        """Send a message to the partner actor."""
        if self.partners:
            for partner in self.partners:
                try:
                    partner.tell(message)
                except pykka.ActorDeadError:
                    print(f"{self.ID} failed to send message to {partner}, the actor is dead.")
        else:
            print(f"{self.ID} has no partners to send the message to.")
            
    def on_stop(self):
        """Clean up when the actor stops."""
        print(f"{self.ID} is stopping.")

    