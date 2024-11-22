from attack.attacker import Attacker
from attack.random_attacker import RandomAttacker
from attack.sign_flipping_attacker import SignFlippingAttacker
from attack.targeted_label_flipping_attacker import TargetedLabelFlippingAttacker
from attack.untargeted_label_flipping_attacker import UntargetedLabelFlippingAttacker
from nodes.mab_malicious_nodes import *


class MabAttacker(Attacker[MabMaliciousNode]):
    pass


class MabRandomAttacker(MabAttacker, RandomAttacker):
    pass


class MabSignFlippingAttacker(MabAttacker, SignFlippingAttacker):
    pass


class MabTargetedLabelFlippingAttacker(MabAttacker,
                                       TargetedLabelFlippingAttacker):
    pass


class MabUntargetedLabelFlippingAttacker(MabAttacker,
                                         UntargetedLabelFlippingAttacker):
    pass
