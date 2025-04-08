from attack.attacker import Attacker
from nodes.sydelp_malicious_nodes import SydelpMaliciousNode


class SydelpAttacker(Attacker[SydelpMaliciousNode]):
    computing_power: int # P

    def __init__(self,
                 computing_power: int,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.computing_power = computing_power

    def update_nodes(self) -> None:
        pass
