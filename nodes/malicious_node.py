from keras.src.models import Model as KerasModel
from nodes.node import Node


class MaliciousNode(Node):
    attacking: bool
    is_malicious: bool = True

    def __init__(self, attacking: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attacking = attacking
