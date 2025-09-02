Welcome to slurp's documentation!
=============================================================================

SLURP is your companion to compute a simple land-use/land-cover mask from Very High Resolution (VHR) optical
images. It proposes different few or unsupervised learning algorithms that produce one-versus-all masks
(water, vegetation, shadow, urban). Then a final algorithm stacks them all together and regularize them to
obtain into a single multiclass mask.

.. toctree::
   :caption: Installation
   :maxdepth: 1

   Installation procedure <installation>

.. toctree::
   :caption: Documentation
   :maxdepth: 1

   SLURP configuration <slurp_config>
   Tutorial <tutorial>
   CLI Usage <usage_cli>
   API Usage <usage_api>
   API Documentation <apidoc/modules>

.. toctree::
   :caption: Contributing
   :maxdepth: 1

   Guide for developers <developer_guide>
   Contribute to slurp ! <contributing>

