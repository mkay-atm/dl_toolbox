#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

### used to generate configuration information ###
class config(object):
    content= {}
    
    def __init__(self, content):
        self.content = content
    
    @staticmethod
    def gen_confDict(arg=True, url= 'C:/Users/mkayser/Documents/Notebooks/wl_testing/wl_44_markus.conf'):
        """reading the config into a dictionary for easy processing at later stages. If no other URL is 
        specified, use the above as default."""
        confFile = [Path(url)]
        for file_name in confFile:
            with open(str(file_name)) as reader:
                    # one step approach, fast but hard to understand
                return {((line.strip('\n')).split('='))[0]: ((line.strip('\n')).split('='))[1].strip('\t')  
                        for line in reader if not line.startswith('#')}
                # return {((line.strip('\n')).split('='))[0]: ((line.strip('\n')).split()[1].strip('\t')  
                #         for line in reader if not line.startswith('#')}
