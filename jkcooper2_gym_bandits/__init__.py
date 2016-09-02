from gym.envs.registration import register
from gym.scoreboard.registration import add_task, add_group
from .package_info import USERNAME

# Env registration
# ==========================
envs = ['BanditTenArmedRandomFixed',
        'BanditTenArmedRandomRandom',
        'BanditTenArmedRandomStochastic',
        'BanditTwoArmedDeterministicFixed',
        'BanditTwoArmedHighHighFixed',
        'BanditTwoArmedHighLowFixed',
        'BanditTwoArmedHighLowFixedNegative',
        'BanditTwoArmedLowLowFixed']

for env in envs:
    register(
        id='{}/{}-v0'.format(USERNAME, env),
        entry_point='{}_gym_bandits:{}'.format(USERNAME, env),
        timestep_limit=1,
        nondeterministic=True,
    )

# Scoreboard registration
# ==========================
add_group(
    id='bandits',
    name='Bandits',
    description='Various N-Armed Bandit environments'
)

for env in envs:
    add_task(
        id='{}/{}-v0'.format(USERNAME, env),
        group='bandits',
        summary='{}'.format(env),
    )
