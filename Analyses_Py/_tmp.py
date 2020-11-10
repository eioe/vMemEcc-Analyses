

evo_s = list()
evo_l = list()
for sub in sub_list_str: 
    evo_s.append(get_epos(sub, 'stimon', 'EccS', event_dict).average())

