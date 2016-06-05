# Quantified Boolean Formulas solver #

A QBF solver using two adversarial neural networks.

  * Setup :
    * ./setup.sh
  * Usage :
    * th qbfs.lua Filename [epochs [sleep [verbosity]]]
    * epochs    : number of game sessions
    * sleep     : number of seconds between each print
    * verbosity :
      * 0 : silent
      * 1 : sessions only
      * 2 : sessions and training sets
      * 3 : full