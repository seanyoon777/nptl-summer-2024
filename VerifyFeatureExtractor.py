from NeuralProcessor import *

N_ARRAYS = 1
N_ELECTRODES = 128 
FS = 30000

diagnostic_NSP1_filename = '/oak/stanford/groups/henderj/braingate/t12/t12.2022.08.13/Data/_NSP1/NSP Data/0_neuralProcess_Complete_bld(000)001.ns5'
diagnostic_NSP2_filename = '/oak/stanford/groups/henderj/braingate/t12/t12.2022.08.13/Data/_NSP2/NSP Data/0_neuralProcess_Complete_bld(000).ns5'
block_NSP1_filename      = '/oak/stanford/groups/henderj/braingate/t12/t12.2022.08.13/Data/_NSP1/NSP Data/1_neuralProcess_Complete_bld(001)002.ns5'
block_NSP2_filename      = '/oak/stanford/groups/henderj/braingate/t12/t12.2022.08.13/Data/_NSP2/NSP Data/1_neuralProcess_Complete_bld(001).ns5'
save_path = '/oak/stanford/groups/henderj/sean777/OptimalNeuralFeatures/Results/'

def main():
    # load diagnostic blocks
    diagnostic_raw1 = read_ns5_file(diagnostic_NSP1_filename, N_ELECTRODES)
    diagnostic_raw2 = read_ns5_file(diagnostic_NSP2_filename, N_ELECTRODES)
    
    # load raw neural blocks
    raw_neural1 = read_ns5_file(block_NSP1_filename)
    raw_neural2 = read_ns5_file(block_NSP2_filename)
    
    # initialize Neuralprocessor for diagnostic block and denoise 
    diagnostic_processor = NeuralProcessor({
        "processes": [
            ChevyshevFilter("lowpass", Wn=[int(0.4 * FS/2)], rp=0.05, ord=8, fs=FS, non_causal=True), 
            Downsampler(ds_factor=2),
            ButterworthFilter("bandpass", Wn=[250, 4900], ord=4, fs=FS, non_causal=True)
        ]
    })
    
    # get neural data from diagnostic blocks before LRR
    diagnostic_neural1 = diagnostic_processor(diagnostic_raw1, verbose=True)["lfp"]
    diagnostic_neural2 = diagnostic_processor(diagnostic_raw2, verbose=True)["lfp"]
    
    # get LRR weights -- this will be reused for processing raw_neural1 and raw_neural2
    LRRFilter1 = ReReferenceFilter("lrr")
    LRRFilter1.set_weights_with(diagnostic_neural1)
    LRRFilter2 = ReReferenceFilter("lrr")
    LRRFilter2.set_weights_with(diagnostic_neural2)
    
    # now apply LRR
    diagnostic_neural1 = LRRFilter1(diagnostic_neural1, N_ARRAYS, N_ELECTRODES, FS)
    diagnostic_neural2 = LRRFilter2(diagnostic_neural2, N_ARRAYS, N_ELECTRODES, FS)
    
    # denoising is completed for diagnostic blocks. use this to get the thresholds
    thresholds1 = ThresholdCrossingExtractor([],0,0,"std").get_raw_threshold(diagnostic_neural1)
    thresholds2 = ThresholdCrossingExtractor([],0,0,"std").get_raw_threshold(diagnostic_neural2)
    
    # use thresholds and LRR weights to initialize Neuralprocessor for block processing
    block1_processor_params = {
        "processes": [
            ChevyshevFilter("lowpass", Wn=[int(0.4 * FS/2)], rp=0.05, ord=8, fs=FS, non_causal=True), 
            Downsampler(ds_factor=2),
            ButterworthFilter("bandpass", Wn=[250, 4900], ord=4, fs=FS, non_causal=True)
        ],
        "thresh_mults": [-3.5, -4.5], 
        "spike_pow_bands": 'full', 
        "bin_size": 20, 
        "shift_size": 20, 
        "fs": 30000
    }
    block2_processor_params = block1_processor_params.copy()
    
    # print block processor parameters for debugging purposes 
    print("Block processor parameters")
    print(block1_processor_params)
    print(block2_processor_params)
    
    # set parameters 
    block1_processor_params["processes"].append(LRRFilter1)
    block2_processor_params["processes"].append(LRRFilter2)
    block1_processor_params["thresh_values"] = thresholds1
    block2_processor_params["thresh_values"] = thresholds2 
    
    # extract features 
    Processor1 = NeuralProcessor(block1_processor_params)
    Processor2 = NeuralProcessor(block2_processor_params)
    out1 = Processor1(raw_neural1, N_ARRAYS, N_ELECTRODES, FS, verbose=True)
    out2 = Processor2(raw_neural2, N_ARRAYS, N_ELECTRODES, FS, verbose=True)
    
    # save results 
    savemat(f"{save_path}/240514_exp1_out1.mat", out1)
    savemat(f"{save_path}/240514_exp1_out2.mat", out2)


if __name__ == '__main__':
    main()