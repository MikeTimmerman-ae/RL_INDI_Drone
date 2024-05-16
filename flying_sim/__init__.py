from gymnasium.envs.registration import register

register(
    id="flying_sim/FlightEnv",
    entry_point="flying_sim.env:FlightEnv",
)
