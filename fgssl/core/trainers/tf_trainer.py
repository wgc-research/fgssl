import tensorflow as tf

import numpy as np
from federatedscope.core.trainers import Trainer
from federatedscope.core.auxiliaries.enums import MODE
from federatedscope.core.auxiliaries.utils import batch_iter
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.auxiliaries.enums import LIFECYCLE


class GeneralTFTrainer(Trainer):
    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = self.hooks_in_train if hooks_set is None else hooks_set

        self.ctx.check_split(target_data_split_name)

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)

        # TODO: The return values should be more flexible? Now: sample_num,
        #  model_para, results={k:v}

        return num_samples, self.ctx.model.state_dict(), self.ctx.eval_metrics

    def parse_data(self, data):
        """Populate "{}_data", "{}_loader" and "num_{}_data" for different
        modes

        """
        init_dict = dict()
        if isinstance(data, dict):
            for mode in ["train", "val", "test"]:
                init_dict["{}_data".format(mode)] = None
                init_dict["{}_loader".format(mode)] = None
                init_dict["num_{}_data".format(mode)] = 0
                if data.get(mode, None) is not None:
                    init_dict["{}_data".format(mode)] = data.get(mode)
                    init_dict["num_{}_data".format(mode)] = len(data.get(mode))
        else:
            raise TypeError("Type of data should be dict.")
        return init_dict

    def register_default_hooks_train(self):
        self.register_hook_in_train(self._hook_on_fit_start_init,
                                    "on_fit_start")
        self.register_hook_in_train(self._hook_on_epoch_start,
                                    "on_epoch_start")
        self.register_hook_in_train(self._hook_on_batch_start_init,
                                    "on_batch_start")
        self.register_hook_in_train(self._hook_on_batch_forward,
                                    "on_batch_forward")
        self.register_hook_in_train(self._hook_on_batch_forward_regularizer,
                                    "on_batch_forward")
        self.register_hook_in_train(self._hook_on_batch_backward,
                                    "on_batch_backward")
        self.register_hook_in_train(self._hook_on_batch_end, "on_batch_end")
        self.register_hook_in_train(self._hook_on_fit_end, "on_fit_end")

    def register_default_hooks_eval(self):
        # test/val
        self.register_hook_in_eval(self._hook_on_fit_start_init,
                                   "on_fit_start")
        self.register_hook_in_eval(self._hook_on_epoch_start, "on_epoch_start")
        self.register_hook_in_eval(self._hook_on_batch_start_init,
                                   "on_batch_start")
        self.register_hook_in_eval(self._hook_on_batch_forward,
                                   "on_batch_forward")
        self.register_hook_in_eval(self._hook_on_batch_end, "on_batch_end")
        self.register_hook_in_eval(self._hook_on_fit_end, "on_fit_end")

    def _hook_on_fit_start_init(self, ctx):
        # prepare model
        ctx.model.to(ctx.device)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_epoch_start(self, ctx):
        # prepare dataloader
        setattr(ctx, "{}_loader".format(ctx.cur_split),
                batch_iter(ctx.get("{}_data".format(ctx.cur_split))))

    def _hook_on_batch_start_init(self, ctx):
        # prepare data batch
        try:
            ctx.data_batch = next(ctx.get("{}_loader".format(ctx.cur_split)))
        except StopIteration:
            raise StopIteration

    def _hook_on_batch_forward(self, ctx):

        ctx.optimizer = ctx.model.optimizer

        ctx.batch_size = len(ctx.data_batch)

        with ctx.model.graph.as_default():
            with ctx.model.sess.as_default():
                feed_dict = {
                    ctx.model.input_x: ctx.data_batch['x'],
                    ctx.model.input_y: ctx.data_batch['y']
                }
                _, batch_loss, y_true, y_prob = ctx.model.sess.run(
                    [
                        ctx.model.train_op, ctx.model.losses,
                        ctx.model.input_y, ctx.model.out
                    ],
                    feed_dict=feed_dict)
                ctx.loss_batch = batch_loss
                ctx.y_true = CtxVar(y_true, LIFECYCLE.BATCH)
                ctx.y_prob = CtxVar(y_prob, LIFECYCLE.BATCH)

    def _hook_on_batch_forward_regularizer(self, ctx):
        pass

    def _hook_on_batch_backward(self, ctx):
        pass

    def _hook_on_batch_end(self, ctx):
        # TODO: the same with the torch_trainer
        # update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

        # cache label for evaluate
        ctx.ys_true.append(ctx.y_true.detach().cpu().numpy())
        ctx.ys_prob.append(ctx.y_prob.detach().cpu().numpy())

    def _hook_on_fit_end(self, ctx):
        """Evaluate metrics.

        """
        ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar(np.concatenate(ctx.ys_prob), LIFECYCLE.ROUTINE)
        results = self.metric_calculator.eval(ctx)
        setattr(ctx, 'eval_metrics', results)

    def update(self, model_parameters, strict=False):
        self.ctx.model.load_state_dict(model_parameters, strict=strict)

    def save_model(self, path, cur_round=-1):
        pass

    def load_model(self, path):
        pass

    def discharge_model(self):
        pass
