# Core abstractions

Base classes that every environment and reward module implements. To add a new
environment, subclass [`BaseEnvironment`][gfnx.base.BaseEnvironment] together
with `BaseEnvState` / `BaseEnvParams`; to add a new reward, subclass
[`BaseRewardModule`][gfnx.base.BaseRewardModule] and `BaseRewardParams`.

::: gfnx.base
    options:
      show_root_heading: false
      members:
        - BaseEnvState
        - BaseEnvParams
        - BaseRewardParams
        - BaseRewardModule
        - BaseEnvironment
        - BaseRenderer

## Spaces

::: gfnx.spaces
    options:
      show_root_heading: false
      heading_level: 3
