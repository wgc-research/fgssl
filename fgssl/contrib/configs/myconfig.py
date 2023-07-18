from federatedscope.core.configs.config import CN


def extend_training_cfg(cfg):
    # ------------------------------------------------------------------------ #
    # Trainer related options
    # ------------------------------------------------------------------------ #
    cfg.data = CN()

    cfg.data.fgcl = False

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_training_cfg)


def assert_training_cfg(cfg):
    if cfg.backend not in ['torch', 'tensorflow']:
        raise ValueError(
             "Value of 'cfg.backend' must be chosen from ['torch', 'tensorflow']."
        )

# from federatedscope.register import register_config
# register_config("fl_training", extend_training_cfg)