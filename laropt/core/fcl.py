
# ROOT/LArSoft specifics
import ROOT
from galleryUtils import loadConfiguration

from laropt.core.parameters import ParamSpec, ParamSet
from typing import Dict

import re

class FclManager:

    def __init__(self, template_path: str):
        with open(template_path) as f:
            self.template = f.read()

    def generate(self, params: ParamSet, specs: Dict[str, ParamSpec], output_path: str, tag: str):
        text = self.template

        text+='\n# ---\n# Custom configuration settings below (tag: {tag})\n# ---\n\n'.format(tag=tag)

        for name, value in params.values.items():
            spec = specs[name]

            for path in spec.fcl_paths:
                if spec.is_per_plane:
                    for i in range(spec.size):
                        full_path = path.format(plane=i)
                        text = self._set_value(text, full_path, value[i])
                else:
                    text = self._set_value(text, path, value)

        with open(output_path, 'w') as f:
            f.write(text)

    def _set_value(self, text, key, value):
        val_str = f'[{", ".join(map(str, value))}]' if isinstance(value, list) else str(value)
        pattern = rf'({re.escape(key)}\s*:\s*)([^\n#]+)'

        if re.search(pattern, text):
            return re.sub(pattern, rf'\1{val_str}', text)
        else:
            return text + f'{key}: {val_str}\n'

    def parse(self, fcl_path: str, specs: Dict[str, ParamSpec]) -> ParamSet:

        configuration = loadConfiguration(fcl_path)

        values = {}

        for name, spec in specs.items():
            if spec.is_per_plane:
                vals = []
                for i in range(spec.size):
                    full_path = spec.fcl_paths[0].format(plane=i)
                    if not configuration.has_key(full_path):
                        print(f'[FCL MANAGER WARNING CRITICAL] Missing key {name} value, this will be assigned a null value -1')
                        vals.append(-1)
                    else:
                        match = configuration.get[spec.kind](full_path)
                        # print(f'[FCL MANAGER DEBUG] For {name} found {match}')
                        vals.append(match)
                values[name] = vals
            else:
                # is either a singleton (size==1) or a tuple
                if spec.size == 1:
                    if not configuration.has_key(spec.fcl_paths[0]):
                        print(f'[FCL MANAGER WARNING CRITICAL] Missing key {name} value, this will be assigned a null value -1')
                        values[name] = -1
                    else:
                        values[name] = configuration.get[spec.kind](spec.fcl_paths[0])
                        # print(f'[FCL MANAGER DEBUG] For {name} found {match}')
                else:
                    if not configuration.has_key(spec.fcl_paths[0]):
                        print(f'[FCL MANAGER WARNING CRITICAL] Missing key {name} value, this will be assigned a null value -1')
                        values[name] = [-1] * spec.size
                    else:
                        values[name] = list(configuration.get[f'std::vector<{spec.kind}>'](spec.fcl_paths[0]))
                        # print(f'[FCL MANAGER DEBUG] For {name} found {match}')

        parameter_set = ParamSet(values)
        print(parameter_set)
        return parameter_set