:orphan:

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

attribute

.. auto{{ objtype }}:: {{ fullname | replace("ntcad.", "ntcad::") }}

{# In the fullname (e.g. `ntcad.omen.operations.methodname`), the module name
is ambiguous. Using a `::` separator (e.g. `ntcad::omen.operations.methodname`)
specifies `ntcad` as the module name. #}