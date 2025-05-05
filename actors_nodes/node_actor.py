"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-12-28 12:37:50
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2025-05-05 15:03:43
FilePath: actors_nodes/node_actor.py
Description: 这是默认设置,可以在设置》工具》File Description中进行配置
"""
import pykka


class NodeActor(pykka.ThreadingActor):
    """
    Node Actor class
    
    *Description*
    
    This class is used to implement the node actor in the network
    
    As node actor we mean an agent in the network, regardless of its role (trainer, attacker, verifier, etc.).
    
    It contains the primary functions to handle the messages and the communication between the actors.
    """
    ID: int
    partners: list
    round: int
    

    def __init__(self, ID: int) -> None:
        super().__init__()
        self.ID = ID
        self.partners = []
        self.round = 0     
    

    def set_partner(self, partners):
        """Set the partner actor."""
        try:
            self.partners = partners
            # print(f"\n\n{self.ID} set partners: {self.partners}\n\n")
        except Exception as e:
            print(f"{self.ID} failed to set partner: {e}")
            return
        
    def update_partners(self, partners):
        # recheck this function
        self.partners = partners

    def on_receive(self, message):
        """Handle incoming messages."""        
        # if the message is not a dictionary, print an error message
        if not isinstance(message, dict):
            # print(f"\n{self.ID} received an invalid message: {message}\n")
            return

        # 1- If the message is a string "stop", stop the actor
        if message.get('command') == 'stop':
            self.stop()
            return
        
        if message.get('command') == 'debug':
            print(f"\n{self.ID} partners: {self.partners}\n")
        
        if message.get('command') == 'is_alive':
            print(f"{self.ID} is alive")
            return

        # 2- If the message is a dictionary with the key "command" and the value "set_partner",
        if message.get('command') == 'set_partners':
            self.set_partner(message['partners'])
            return

        # 3- If the message is a dictionary with the key "command" and the value "update_partners",
        if message.get('command') == 'update_partners':
            self.update_partners(message['partners'])
            # print(f"{self.ID} updated partners to {message['partners']}")
            return
        
        # NORMAL MESSAGE HANDLING are in the subclasses
        

    def broadcast(self, message):
        """Send a message to all the partners."""
        if self.partners:
            for partner in self.partners:
                try:
                    partner.tell(message)
                except pykka.ActorDeadError:
                    pass
                    # print(f"{self.ID} failed to send message to {partner}, the actor is dead.")
        else: 
            print(f"{self.ID} has no partners to send the message to.")
            
    def on_stop(self):
        """Clean up when the actor stops."""
        # print(f"{self.ID} is stopping.")

    