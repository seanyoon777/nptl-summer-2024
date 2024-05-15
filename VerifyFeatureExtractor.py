# This code verifies whether 
from NeuralProcessor import *

FS = 30000

diagnostic_NSP1_filename = ''
diagnostic_NSP2_filename = '/oak/stanford/groups/henderj/braingate/t12/t12.2022.08.13/Data/_NSP2/NSP Data/0_neuralProcess_Complete_bld(000).ns5'
block_NSP1_filename = ''
block_NSP2_filename = ''

def main():
    # load diagnostic blocks
    diagnostic_raw1 = read_ns5_file(diagnostic_NSP1_filename)
    diagnostic_raw2 = read_ns5_file(diagnostic_NSP2_filename)
    
    # load raw neural blocks
    raw_neural1 = read_ns5_file(block_NSP1_filename)
    raw_neural2 = read_ns5_file(block_NSP2_filename)
    
    # initialize Neuralprocessor for diagnostic block and denoise 
    diagnostic_processor = NeuralProcessor({
        "processes": [
            ChevyshevFilter("lowpass", Wn=[int(FS * 0.2)], rp=0.05, ord=8, fs=FS, non_causal=True), 
            Downsampler(ds_factor=2),
            ButterworthFilter("bandpass", Wn=[250, 4900], ord=4, fs=FS, non_causal=True)
        ], 
        
    })
    diagnostic_neural1 = diagnostic_processor(diagnostic_raw1)["lfp"]
    diagnostic_neural2 = diagnostic_processor(diagnostic_raw2)["lfp"]
    
    # get LRR weights
    LRRFilter1 = ReReferenceFilter("lrr")
    LRRFilter1.set_weights_with(diagnostic_neural1)
    
    LRRFilter2 = ReReferenceFilter("lrr")
    LRRFilter2.set_weights_with(diagnostic_neural2)
    
    # get raw thresholds 
    thresholds1 = ThresholdCrossingExtractor([],0,0,"std").get_raw_threshold(diagnostic_neural1)
    thresholds2 = ThresholdCrossingExtractor([],0,0,"std").get_raw_threshold(diagnostic_neural2)
    
    #TODO: 
    # - Implement get_weights and set_weights_with 
    
    

    # So initialization: 
    #   

    
    # get threshold 
    # initialize Neuralprocessor for actual data block and denoise 
    block1_processor_params = {
        "processes": [
            ChevyshevFilter("lowpass", Wn=[6000], rp=0.05, ord=8, fs=30000, non_causal=True), 
            Downsampler(ds_factor=2),
            ButterworthFilter("bandpass", Wn=[250, 4900], ord=4, fs=30000, non_causal=True)
        ], 
        "thresh_mults": [-3.5, -4.5], 
        "spike_pow_bands": 'full', 
        "bin_size": 20, 
        "shift_size": 20, 
        "fs": 30000
    }
    block2_processor_params = block1_processor_params.copy() 
    
    block1_processor_params["thresh_values"] = std_mat1 
    block2_processor_params["thresh_values"] = std_mat2
    
    block1_processor_params["processes"] = std_mat1 
    block2_processor_params["processes"] = std_mat2
    
    
    
            "thresh_values": std_mat1,
        "threshold_multipliers": [-3.5, -4.0, -4.5],  # threshold multipliers for threshold crossing count – Optional
    "spike_pow_bands": [(10, 25), (25, 40), (40, 65), (65, 125), (125, 250), (125, 500)],  # in Hz     – Optional 
    "thresh_method": "rms",     # rms or std supported, but rms preferred                              – Optional. Only required if extracting threshold crossing count
    "lmp_boxsizes": [50, 100],  # local motor potential boxsize, in ms                                 – Optional. Only required if extracting local motor potential
    "bin_size": 20,             # bin size for spiking bandpower and threshold crossing count, in ms   – Optional. Only required if extracting threshold crossing count or spike bandpower
    "shift_size": 20,           # bin shift for spiking bandpower and threshold crossing count, in ms  – Optional. Only required if extracting threshold crossing count or spike bandpower
    "fs": 30000                 # signal sampling frequency in Hz
    block_processor_params["processes"].append(ReReferenceFilter("lrr", ref_mat=ref_mat1))
                
    
    = NeuralProcessor({
        "processes": [
            ChevyshevFilter("lowpass", Wn=[6000], rp=0.05, ord=8, fs=30000, non_causal=True), 
            Downsampler(ds_factor=2),
            ButterworthFilter("bandpass", Wn=[250, 4900], ord=4, fs=30000, non_causal=True), 
            ReReferenceFilter("lrr", ref_mat=ref_mat1)
        ]
    })

    


processor_params = {
    "processes": [
        ChevyshevFilter("lowpass", Wn=[6000], rp=0.05, ord=8, fs=30000, non_causal=True),
        Downsampler(ds_factor=2), 
        ButterworthFilter("bandpass", Wn=[250, 4900], ord=4, fs=30000, non_causal=True), 
        ReReferenceFilter("lrr", max_seconds=60)
    ], 
    "thresh_mults": [-4.5], 
    "thresh_method": "std",
    "spike_pow_bands": ['full'], 
    "bin_size": 20,             # bin size for spiking bandpower and threshold crossing count, in ms   – Optional. Only required if extracting threshold crossing count or spike bandpower
    "shift_size": 20,           # bin shift for spiking bandpower and threshold crossing count, in ms  – Optional. Only required if extracting threshold crossing count or spike bandpower
}


if __name__ == '__main__':
    main()