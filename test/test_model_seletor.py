import context
from core.model_seletor import Model_seletor

model_base = '../core/model/'
rule_name = "color_reproduction_delay_unit"

ms = Model_seletor()
ms.delete(model_base + rule_name)
