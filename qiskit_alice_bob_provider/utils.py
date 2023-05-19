##############################################################################
# Copyright 2023 Alice & Bob
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

import re

_CAMEL_PATTERN = re.compile(r'(?<!^)(?=[A-Z])')


def camel_to_snake_case(name: str) -> str:
    return _CAMEL_PATTERN.sub('_', name).lower()


def snake_to_camel_case(name: str) -> str:
    upper_camel = ''.join(x.capitalize() for x in name.lower().split('_'))
    return upper_camel[0].lower() + upper_camel[1:]
