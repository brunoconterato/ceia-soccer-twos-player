"""
Script usado para corrigir pacotes que não podem ser carregados pelo Pickle se abrir em, outro módulo.

IMPORTANTE: Necessita ser executado a partir do arquivo init.py na raiz do projeto! (chamando o método fix_pickle)
IMPORTANTE 2: Requer cloudpickle >= 2.0.0 para ffuncionar. Mas o ambiente até então precisava da versão 1.6.0.
    Então é necessário atualizar o cloudpickle para rodar esse script, depois pode ser necessário retorná-lo para a versão 1.6.0.

O Erro ocorreu porque o treino foi executado a partir do init.py na raiz do projeto,
mas o agente não será executado de lá (será executado de algum outro local),
então nenhum módulo dentro de src poderá ser encontrado pelo agente sem executar este script.


------------
------------

(SOMENTE se estiver trabalhando com treinos gerados a partir de init.py:)
IMPORTANTE: é preciso renomear os imports para:
from .utils import create_custom_env
from .config import ENVIRONMENT_ID, config
from .stop import stop
Pode ser preciso também usar python 3.7

------------
------------

"""

import pickle5 as pickle # python == 3.7
# import pickle # Se Python >= 3.8
import cloudpickle
import src
import src.experiments.ppo_multiagent.callback as callback
# import callback

def register_modules():
    cloudpickle.register_pickle_by_value(src)
    cloudpickle.register_pickle_by_value(callback)

def fix_pickle(file_with_error = None):
    if file_with_error is None:
        return

    register_modules()
    with open(file_with_error, "rb") as f:
        config = pickle.load(f)
    with open(file_with_error, 'wb') as f:
        cloudpickle.dump(config, f)

if __name__ == "__main__":

    # error_config = "/home/bruno/Workspace/soccer-tows-player/src/experiments/ppo_multiagent/agents/src/my_ray_soccer_agent/Testing_env/PPO_Soccer_6d320_00000_0_2021-11-26_22-36-22/checkpoint_000001/../params.pkl"
    file_with_error = "/home/bruno/Workspace/soccer-tows-player/src/ray_results/Testing_env/PPO_Soccer_18d23_00000_0_2021-11-24_20-34-41/params.pkl"
    # file_with_error = "/home/bruno/Workspace/soccer-tows-player/src/ray_results/Testing_pickle/PPO_Soccer_dd25d_00000_0_2021-11-28_19-16-42/params.pkl"

    fix_pickle(file_with_error)