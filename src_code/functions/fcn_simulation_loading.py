
"""
functions for loading simulations
"""


def fcn_set_sweepParam_string(n_sweepParams, sweep_param_name, swept_params_dict, param_indx):
    
    param_var_name = 'param_vals1'
    param_val = swept_params_dict[param_var_name][param_indx]
    sweep_param_str = ( '%s%0.3f' % (sweep_param_name, param_val) )
    
    for i in range(1, n_sweepParams):
    
        param_var_name = ('param_vals%d' % (i+1))
        param_val = swept_params_dict[param_var_name][param_indx]
        param_str_piece = ('_%0.3f' % param_val)
        sweep_param_str = sweep_param_str + param_str_piece
                
    return sweep_param_str
