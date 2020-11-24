

evo_s = list()
evo_m = list()
evo_l = list()
for sub in sub_list_str: 
    evo_s.append(get_epos(sub, 'stimon', 'LoadLow', event_dict).average())
    evo_m.append(get_epos(sub, 'stimon', 'LoadHigh', event_dict).average())
    evo_l.append(get_epos(sub, 'stimon', 'EccL', event_dict).average())
