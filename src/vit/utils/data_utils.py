import torch
import torch.nn.functional as F
from quant.quant_model import set_quant_state

def save_inp_oup_data(model, layer , cali_data,
                      batch_size: int = 32, keep_gpu: bool = True,
                      input_prob: bool = False):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param weight_quant: use weight_quant quantization
    :param act_quant: use act_quant quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device, input_prob=input_prob)
    cached_batches = []

    for i in range(int(cali_data.size(0) / batch_size)):
        if input_prob:
            cur_inp, cur_out, cur_sym = get_inp_out(cali_data[i * batch_size:(i + 1) * batch_size])
            cached_batches.append((cur_inp.cpu(), cur_out.cpu(), cur_sym.cpu()))
        else:
            cur_inp, cur_out = get_inp_out(cali_data[i * batch_size:(i + 1) * batch_size])
            cached_batches.append((cur_inp.cpu(), cur_out.cpu()))
    cached_inps = torch.cat([x[0] for x in cached_batches])
    cached_outs = torch.cat([x[1] for x in cached_batches])
    if input_prob:
        cached_sym = torch.cat([x[2] for x in cached_batches])
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
        if input_prob:
            cached_sym = cached_sym.to(device)
    if input_prob:
        return (cached_inps, cached_sym), cached_outs
    return (cached_inps,), cached_outs


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    def __init__(self, model, layer,
                 device: torch.device, input_prob: bool = False):
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)
        self.input_prob = input_prob

    def __call__(self, model_input):
        set_quant_state(model = self.model, weight_quant = False, act_quant = False)

        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                _ = self.model(model_input.to(self.device))
            except StopForwardException:
                pass
            if self.input_prob:
                input_sym = self.data_saver.input_store[0].detach()
            
            # Recalculate input with network quantized
            self.data_saver.store_output = False
            set_quant_state(model = self.model, weight_quant = False, act_quant = True)
            try:
                _ = self.model(model_input.to(self.device))
            except StopForwardException:
                pass

            self.data_saver.store_output = True
        handle.remove()

        if self.input_prob:
            return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach(), input_sym
        return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()



