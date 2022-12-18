import torch
import torch.nn
from torch.utils.checkpoint import checkpoint

class UnicronBase(torch.nn.Module):

    def __init__(self, levels, in_channels, base_channels, out_channels, **level_kwargs):
        super(UnicronBase, self).__init__()

        self.levels = levels

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels

        self.level_kwargs = level_kwargs

        self.input_map = self.assemble_input_map()
        self.cycle = self.assemble_cycle()
        self.output_map = self.assemble_output_map()

    def assemble_input_map(self):
        raise NotImplementedError()

    def assemble_cycle(self):
        raise NotImplementedError()

    def assemble_output_map(self):
        raise NotImplementedError()

    def forward(self, input):

        x_f = self.input_map(input)
        y_f = self.cycle(x_f)
        output = self.output_map(y_f)

        return output


class UnicronLevelBase(torch.nn.Module):

    def __init__(self, max_levels, level, base_channels, checkpointing=True):

        super(UnicronLevelBase, self).__init__()

        self.max_levels = max_levels
        self.level = level
        self.checkpointing = checkpointing
        self.base_channels = base_channels

        self.finest = (self.level == 0)
        self.coarsest = (self.level == self.max_levels-1)

        if self.coarsest:
            self.megatron_coarsest_smooth = self.assemble_megatron_coarsest_smooth()
        else:
            if self.finest:
                self.megatron_pre_smooth = self.assemble_megatron_finest_pre_smooth()
            else:
                self.megatron_pre_smooth = self.assemble_megatron_pre_smooth()
            self.restriction = self.assemble_restriction()
            self.sublevel = self.assemble_sublevel()
            self.prolongation = self.assemble_prolongation()
            self.correction = self.assemble_correction()
            self.megatron_post_smooth = self.assemble_megatron_post_smooth()

    def channels(self, level=None):
        if level is None:
            level = self.level
        return (2**level)*self.base_channels

    def assemble_megatron_coarsest_smooth(self):
        raise NotImplementedError()

    def assemble_megatron_finest_pre_smooth(self):
        raise NotImplementedError()

    def assemble_megatron_pre_smooth(self):
        raise NotImplementedError()

    def assemble_megatron_post_smooth(self):
        raise NotImplementedError()

    def assemble_restriction(self):
        raise NotImplementedError()

    def assemble_prolongation(self):
        raise NotImplementedError()

    def assemble_correction(self):
        raise NotImplementedError()

    def instantiate_sublevel(self):
        raise NotImplementedError()

    def assemble_sublevel(self):

        # If this level is less than one less than the max, it is a coarsest level
        if not self.coarsest:
            return self.instantiate_sublevel()
        else:
            raise Exception()

    def forward(self, x_f):

        # Coarsest smoothing
        if self.coarsest:
            if self.checkpointing:
                y_f = checkpoint(self.megatron_coarsest_smooth, x_f)
            else:
                y_f = self.megatron_coarsest_smooth(x_f)
            return y_f

        # Presmooting
        if self.checkpointing and not self.finest:
            y_f = checkpoint(self.megatron_pre_smooth, x_f)
        else:
            y_f = self.megatron_pre_smooth(x_f)

        # Restriction
        y_c = self.restriction(y_f)

        # Next level
        y_c = self.sublevel(y_c)

        # Prolongation and correction
        y_c = self.prolongation(y_c)
        y_f = self.correction((y_f, y_c))

        # Post-smoothing
        if self.checkpointing:
            y_f = checkpoint(self.megatron_post_smooth, y_f)
        else:
            y_f = self.megatron_post_smooth(y_f)

        return y_f