# tf-explorer
A command-line interface tool to explore [TensorFlow checkpoints](https://www.tensorflow.org/guide/checkpoints).

`tf-explorer` presents a hierarchical directory structure for variables in your checkpoint. You can
list (`ls`), rename (`mv`), and print (`cat`) stored variables and scopes. And, of course, you
can navigate (`cd`) through the scope hierarchy.

All changes to variables and scopes are kept in-memory until you're ready to commit them back to
your checkpoint with `commit`. As a matter of precaution, I recommend keeping a backup of your
original checkpoint before committing any changes to it.

## Tutorial
The following example opens the latest checkpoint in `/path/to/checkpoint_dir`, prints the
current value of the `global_step` variable, moves it into the `foo/bar/` scope, and renames
the variable to `old_global_step`.

```
  $ ./tf-explorer.sh /path/to/checkpoint_dir
```
```
  Checkpoint loaded from /path/to/checkpoint_dir/model.ckpt-106901.
  Type help or ? to list commands.

  > ls
  beta1_power
  beta2_power
  @conditioning/
  global_step
  @student/
  @teacher/
  > cat global_step
  106901
  > mv global_step foo/bar/old_global_step
```

The @ symbol indicates that the scope is "pure", i.e. it isn't both a variable name and
a scope name. You cannot print out a pure scope since it doesn't store a value.

At this point, the changes to `global_step` are in-memory only. If you want to view which
changes are in-memory but haven't been written to disk, use the `mutations` command:

```
  > mutations
  global_step -> foo/bar/old_global_step
```

If you're happy with these changes, write them back to disk:

```
  > commit
```

Otherwise, you can either continue exploring / making further changes or exit `tf-explorer`:

```
  > exit
  WARNING: there are pending mutations that have not been written to disk. Discard (y/N)? y
```

## License
Apache 2.0
