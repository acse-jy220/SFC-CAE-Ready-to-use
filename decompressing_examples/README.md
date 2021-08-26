## Decompressing Examples
### Reconstructing snapshots from latent variables. (of course error introduced)

* CO2 (337Kb compressed data + 150MB model_dict/ sfcs -> 5.96GB raw data)
```sh
$ bash decompressing_CO2.sh 
```
reconstructed vtu files will be in `reconstrcuted_CO2_letent_4`.

* Slugflow (1.65Mb compressed data + 1.03GB model_dict/ sfcs -> 132GB vtu files)
```sh
$ bash decompressing_slugflow.sh
```
reconstructed vtu files wiil be in `reconstrcuted_slugflow_letent_64`.

If you are interested in how to compress/decompress on other unseen vtu datasets, please have a look in the python scripts as well as the source code in [utils.py](https://github.com/acse-jy220/SFC-CAE-Ready-to-use/blob/main/sfc_cae/utils.py)