# astro_toolbox
## attr
I don't use attr.init to give the basic metadata to each column because new rows may be created without calling the init.
The `.name` attribute of columns is independent of `meta` and need to be handled independently.
If in the future I want to switch between astropy and pandas, I may write an attr.get function to change the basic method without changing other codes.
I don't create a new class, because:
- Sometimes the modifying of the Table/DF will create a new object and destroy what I have created.
- Both Table and DF cannot recall the parent object within the columns, thus it is unconvenient when plotting because the information from the parent table cannt be read.

Reload of the class DataSeries will disrupt the type check already loaded.

## plot
Using too much `kwargs` will make the program less readable.

For functions of "outer" level which has @set_plot decorated function inside:
    - `select` should not be set at the beginning of the function, but should be set in the inner function,
       for `x = x[select]` will destroy the dimension of `x` fed into the inner ones.
    - Sometimes the filename should be set manually, except when the name from the inner function is good enough.

