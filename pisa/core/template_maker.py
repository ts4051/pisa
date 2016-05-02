#! /usr/bin/env python
# authors: J.Lanfranchi/P.Eller
# date:   March 20, 2016

import importlib

from pisa.core.pipeline import Pipeline
from pisa.core.param import ParamSet
from pisa.utils.log import logging, set_verbosity


class TemplateMaker(object):
    """Creates a TemplateMaker object that has several pipelines inside"""
    def __init__(self, pipeline_settings):
        self.pipelines = [Pipeline(setting) for setting in pipeline_settings]

    def compute_outputs(self, **kwargs):
        output = None
        for pipeline in self.pipelines:
            pipeline_output = pipeline.compute_outputs(**kwargs)
            if output is None:
                output = pipeline_output
            else:
                # add together
                if isinstance(pipeline_output, list):
                    output = [sum(x) for x in zip(output, pipeline_output)]
                else: 
                    output += pipeline_output
        return output

    def update_params(self, params):
        [stage.params.update_existing(params) for stage in self]

    @property
    def params(self):
        params = ParamSet()
        [params.extend(pipeline.params) for pipeline in self.pipelines]
        return params

    def update_params(self, paramset):
        for pipeline in self.pipelines:
            pipeline.params.update_existing(paramset)

    @property
    def free_params_values(self):
        return [p.value for p in self.params.free]

    @property
    def free_params_names(self):
        return [p.name for p in self.params.free]

    def set_free_params(self, values):
        for name, value in zip(self.free_params_names, values):
            for pipeline in self.pipeline:
                if name in pipeline.params.free:
                    pipeline.params.free.value = value


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import numpy as np
    from pisa.utils.fileio import from_file, to_file
    from pisa.utils.parse_config import parse_config

    parser = ArgumentParser()
    parser.add_argument('-t', '--template-settings', type=str,
                        metavar='configfile', required=True,
                        help='''settings for the template generation''')
    parser.add_argument('-o', '--outfile', dest='outfile', metavar='FILE',
                        type=str, action='store', default="out.json",
                        help='file to store the output')
    args = parser.parse_args()

    template_config = parse_config(from_file(args.template_settings))

    template_nu_pipeline = Pipeline(template_config)
    m0 = template_nu_pipeline.compute_outputs()
    fp = template_nu_pipeline.params.free #free_params
    fp['test'].value*=1.2
    pipeline.update_params(fp)
    m1 = pipeline.compute_outputs()
    print (m1/m0)['nue'][0,0]
