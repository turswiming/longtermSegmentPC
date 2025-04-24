from .opticalflow_loss_2d import OpticalFlowLoss_2d
from .opticalflow_loss_3d import OpticalFlowLoss_3d
from .opticalflow_loss_formula2 import OpticalFlowLossFormula2
from .trajectory_loss import TrajectoryLoss
from .trajectory_loss_formula3 import TrajectoryLossFormula3
import torch


class CriterionDict:
    def __init__(self, dict):
        self.criterions = dict

    def __call__(self, sample, flow, masks_softmaxed, iteration, train=True, prefix=''):
        loss = torch.tensor(0., device=masks_softmaxed.device, dtype=masks_softmaxed.dtype)
        log_dict = {}
        for name_i, (criterion_i, loss_multiplier_i, anneal_fn_i) in self.criterions.items():
            loss_i = loss_multiplier_i * anneal_fn_i(iteration) * criterion_i(sample, flow, masks_softmaxed, iteration, train=train)
            loss += loss_i
            log_dict[f'loss_{name_i}'] = loss_i.item()

        log_dict['loss_total'] = loss.item()
        return loss, log_dict

    def flow_reconstruction(self, sample, flow, masks_softmaxed):
        if  'opticalflow' in self.criterions:
            return self.criterions['opticalflow'][0].rec_flow(sample, flow, masks_softmaxed)
        raise NotImplementedError
    def process_flow(self, sample, flow):
        if  'opticalflow' in self.criterions:
            return self.criterions['opticalflow'][0].process_flow(sample, flow)
        raise NotImplementedError

    def viz_flow(self, flow):
        if  'opticalflow' in self.criterions:
            return self.criterions['opticalflow'][0].viz_flow(flow)
        raise NotImplementedError

