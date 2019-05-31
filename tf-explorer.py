#!/usr/bin/env python

import cmd
import getopt
import numpy as np
import os
import re
import subprocess
import sys
import tempfile


EDITOR = os.environ.get('EDITOR', 'vim')


class Node:
  def __init__(self, parent, name, is_terminal):
    self.parent = parent or self
    self.name = name
    self.children = []
    self.is_terminal = is_terminal

  def child(self, name):
    for n in self.children:
      if n.name == name:
        return n
    return None

  @property
  def _root(self):
    while self.parent != self:
      self = self.parent
    return self

  def insert(self, name, is_terminal):
    node = self
    components = name.split('/')
    if len(name) > 0 and name[0] == '/':
      node = node._root
    for elem in components:
      if elem == '' or elem == '.':
        pass
      elif elem == '..':
        node = node.parent
      else:
        n = node.child(elem)
        if n is None:
          # Interior nodes are non-terminal
          n = Node(node, elem, is_terminal=False)
          node.children.append(n)
        node = n
    node.is_terminal = is_terminal
    return node

  def find(self, name):
    node = self
    components = name.split('/')
    if len(name) > 0 and name[0] == '/':
      node = node._root
    for elem in components:
      if elem == '' or elem == '.':
        pass
      elif elem == '..':
        node = node.parent
      else:
        node = node.child(elem)
      if node is None:
        return None
    return node

  def find_terminal_nodes(self):
    all_terminal_nodes = []
    to_visit = [self]
    while len(to_visit) > 0:
      node = to_visit.pop(0)
      if node.is_terminal:
        all_terminal_nodes.append(node)
      to_visit += node.children
    return all_terminal_nodes

  def move(self, src, dest):
    def find_terminal_names(node):
      ret = []
      def recurse(node):
        if node.is_terminal:
          ret.append(node.full_name)
        for child in node.children:
          recurse(child)
      recurse(node)
      return ret

    original_names = find_terminal_names(src)

    # Remove last path component to compute `dest_dir`
    components = dest.split('/')
    dest_dir = '/'.join(components[:-1])
    dest_dir = self.insert(dest_dir, is_terminal=False)
    if components[-1] != '':
      if dest_dir.find(components[-1]):
        return None
      src.name = components[-1]

    # Unlink child from parent
    src.parent.children.remove(src)

    # Clean up dangling non-terminal nodes created by removing last child
    parent = src.parent
    while not parent.is_terminal and not parent.is_directory:
      parent.parent.children.remove(parent)
      parent = parent.parent

    # Finish rest of relinking source to destination
    src.parent = None
    dest_dir.children.append(src)
    src.parent = dest_dir

    final_names = find_terminal_names(src)
    return list(zip(original_names, final_names))

  @property
  def full_name(self):
    node = self
    elems = []
    while node != node.parent:
      elems = [node.name] + elems
      node = node.parent
    return '/'.join(elems)

  @property
  def is_directory(self):
    return len(self.children) > 0


class ExplorerShell(cmd.Cmd):
  intro = 'Type help or ? to list commands.\n'
  prompt = '> '
  file = None

  # Hack to handle CTRL+C. Shamelessly stolen from StackOverflow:
  # https://stackoverflow.com/questions/8813291/better-handling-of-keyboardinterrupt-in-cmd-cmd-command-line-interpreter
  def cmdloop(self, intro=None):
    print(self.intro)
    while True:
      try:
        super(ExplorerShell, self).cmdloop(intro='')
        break
      except KeyboardInterrupt:
        print('^C')

  def default(self, line):
    print('{}: unknown command.'.format(line.split()[0]))

  def __init__(self, checkpoint_dir):
    super(ExplorerShell, self).__init__()
    checkpoint_filename = checkpoint_dir
    if tf.gfile.IsDirectory(checkpoint_dir):
      checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
      if checkpoint_state is None:
        print('Checkpoint file not found in {}.'.format(checkpoint_dir))
        sys.exit(1)
      checkpoint_filename = checkpoint_state.model_checkpoint_path
    try:
      tf.train.load_checkpoint(checkpoint_filename)
    except ValueError:
      print('Unable to load checkpoint from {}.'.format(checkpoint_filename))
      sys.exit(1)
    print('Checkpoint loaded from {}.'.format(checkpoint_filename))

    self._checkpoint = checkpoint_filename
    self._root = self._build_tree(self._all_vars())
    self._cwd = self._root
    self._prevwd = '/'
    self._renames = {}
    self._loads = {}

  def help_ls(self):
    print('ls - lists directory contents.')
    print('Syntax: ls [PATH]')

  def complete_ls(self, text, line, begidx, endidx):
    args = line.split()
    if len(args) > 2:
      return []
    text = args[-1] if len(args) == 2 else ''
    path = text.split('/')
    if len(path) == 0:
      node = self._cwd
      filt = ''
    else:
      filt = path[-1]
      path[-1] = ''
      node = self._cwd.find('/'.join(path))
    return [n.name + ('/' if n.is_directory else '') for n in node.children if n.name.startswith(filt)]

  def do_ls(self, arg):
    arg = arg.split()
    if len(arg) == 1:
      target = self._cwd.find(arg[0])
      if target is None:
        print('{}: not found.'.format(arg[0]))
        return
      if not target.is_directory:
        print('/{}'.format(target.full_name))
        return
    elif len(arg) > 1:
      print('ls only supports a single optional argument.')
      return
    else:
      target = self._cwd

    for child in sorted(target.children, key=lambda x: x.name):
      name = child.name
      if not child.is_terminal:
        name = '@' + name
      if child.is_directory:
        name += '/'
      print(name)

  def help_tensors(self):
    print('tensors - lists all tensors in checkpoint.')
    print('Syntax: tensors')

  def do_tensors(self, arg):
    for name, shape in tf.train.list_variables(self._checkpoint):
      print('{} : [{}]'.format(name, ', '.join(map(str, shape))))

  def help_cd(self):
    print('cd - change directory.')
    print('Syntax: cd DIR')

  def complete_cd(self, text, line, begidx, endidx):
    args = line.split()
    if len(args) > 2:
      return []
    text = args[-1] if len(args) == 2 else ''
    path = text.split('/')
    if len(path) == 0:
      node = self._cwd
      filt = ''
    else:
      filt = path[-1]
      path[-1] = ''
      node = self._cwd.find('/'.join(path))
    return [n.name + '/' for n in node.children if n.is_directory and n.name.startswith(filt)]

  def do_cd(self, arg):
    arg = arg.split()
    if len(arg) != 1:
      print('cd: invalid usage.')
      return
    if arg[0] == '-':
      arg[0] = self._prevwd
    target = self._cwd.find(arg[0])
    if target is None:
      print('{}: not found.'.format(arg[0]))
    elif not target.is_directory:
      print('{}: not a directory'.format(arg[0]))
    elif self._cwd != target:
      self._prevwd = '/' + self._cwd.full_name
      self._cwd = target

  def help_pwd(self):
    print('pwd - print working directory.')
    print('Syntax: pwd')

  def do_pwd(self, arg):
    print('/{}'.format(self._cwd.full_name))

  def help_shape(self):
    print('shape - print shape of tensor to console.')
    print('Syntax: shape TENSOR')

  def complete_shape(self, text, line, begidx, endidx):
    return self.complete_cat(text, line, begidx, endidx)

  def do_shape(self, arg):
    arg = arg.split()
    if len(arg) != 1:
      print('shape: invalid usage.')
      return
    target = self._cwd.find(arg[0])
    if target is None:
      print('{}: not found.'.format(arg[0]))
    elif not target.is_terminal:
      print('{}: not a tensor.'.format(arg[0]))
    else:
      # If the tensor was renamed but not committed, find the original name so we can look it up
      # in the checkpoint file.
      name = self._original_name(target.full_name)
      tensor = tf.train.load_variable(self._checkpoint, name)
      if isinstance(tensor, bytes):
        print('[]')
      else:
        print(list(tensor.shape))

  def help_parameters(self):
    print('parameters - print the number of training parameters under a scope.')
    print('Syntax: parameters [PATH]')
    print('Note: the parameter count excludes `Adam` optimizer variables.')

  def complete_parameters(self, text, line, begidx, endidx):
    return self.complete_cat(text, line, begidx, endidx)

  def do_parameters(self, arg):
    arg = arg.split()
    if len(arg) > 1:
      print('parameters: invalid usage.')
    if len(arg) == 0:
      target = self._cwd
    else:
      target = self._cwd.find(arg[0])
      if target is None:
        print('{}: not found.'.format(arg[0]))
        return
    target_names = [node.full_name for node in target.find_terminal_nodes()]
    reader = tf.train.load_checkpoint(self._checkpoint)
    var_shape_map = reader.get_variable_to_shape_map()
    count = 0
    for name in var_shape_map:
      if 'Adam' not in name and name in target_names:
        count += int(np.prod(var_shape_map[name]))
    print('{:,} parameters.'.format(count))

  def help_cat(self):
    print('cat - print tensor to console.')
    print('Syntax: cat TENSOR')

  def complete_cat(self, text, line, begidx, endidx):
    args = line.split()
    if len(args) > 2:
      return []
    text = args[-1] if len(args) == 2 else ''
    path = text.split('/')
    if len(path) == 0:
      node = self._cwd
      filt = ''
    else:
      filt = path[-1]
      path[-1] = ''
      node = self._cwd.find('/'.join(path))
    return [n.name + ('/' if n.is_directory else '') for n in node.children if n.name.startswith(filt)]

  def do_cat(self, arg):
    arg = arg.split()
    if len(arg) != 1:
      print('cat: invalid usage.')
      return
    target = self._cwd.find(arg[0])
    if target is None:
      print('{}: not found.'.format(arg[0]))
    elif not target.is_terminal:
      print('{}: not a tensor.'.format(arg[0]))
    else:
      # If the tensor was renamed but not committed, find the original name so we can look it up
      # in the checkpoint file.
      name = target.full_name
      if name in self._loads:
        value = self._loads[name]
        # If numpy byte string, convert to a decoded Python string for printing.
        if value.dtype.kind == 'S':
          value = value.tostring().decode()
      else:
        value = tf.train.load_variable(self._checkpoint, self._original_name(name))
      if isinstance(value, bytes):
        print(value.decode())
      else:
        print(value)

  def help_save(self):
    print('save - save tensor to disk as numpy array.')
    print('Syntax: save TENSOR FILENAME')

  def do_save(self, arg):
    arg = arg.split()
    if len(arg) != 2:
      print('save: invalid usage.')
      return
    target = self._cwd.find(arg[0])
    if target is None:
      print('{}: not found.'.format(arg[0]))
    elif not target.is_terminal:
      print('{}: not a tensor.'.format(arg[0]))
    else:
      # If the tensor was renamed but not committed, find the original name so we can look it up
      # in the checkpoint file.
      name = self._original_name(target.full_name)
      tensor = tf.train.load_variable(self._checkpoint, name)
      try:
        np.save(arg[1], tensor, allow_pickle=False)
      except Exception as e:
        print(str(e))

  def help_load(self):
    print('load - loads a numpy tensor from disk into the current checkpoint.')
    print('Syntax: load TENSOR FILENAME')
    print('Note: the operation is performed in-memory. To write changes back to the checkpoint, run `commit` after `load`.')

  def do_load(self, arg):
    arg = arg.split()
    if len(arg) != 2:
      print('load: invalid usage.')
      return
    try:
      value = np.load(arg[1], allow_pickle=False)
    except Exception as e:
      print(str(e))
      return
    target = self._cwd.find(arg[0])
    if target is None:
      target = self._cwd.insert(arg[0], is_terminal=True)
    elif not target.is_terminal:
      target.is_terminal = True
    name = target.full_name
    self._loads[name] = value

  def help_zero(self):
    print('zero - zeros out a tensor while retaining its shape.')
    print('Syntax: zero TENSOR')
    print('Note: the operation is performed in-memory. To write changes back to the checkpoint, run `commit` after `zero`.')

  def complete_zero(self, text, line, begidx, endidx):
    return self.complete_cat(text, line, begidx, endidx)

  def do_zero(self, arg):
    arg = arg.split()
    if len(arg) != 1:
      print('zero: invalid usage.')
      return
    target = self._cwd.find(arg[0])
    if target is None:
      print('{}: not found.'.format(arg[0]))
    elif not target.is_terminal:
      print('{}: not a tensor.'.format(arg[0]))
    else:
      # If the tensor was renamed but not committed, find the original name so we can look it up
      # in the checkpoint file.
      name = target.full_name
      tensor = tf.train.load_variable(self._checkpoint, self._original_name(name))
      self._loads[name] = np.zeros_like(tensor)

  def help_edit(self):
    print('edit - launches $EDITOR to allow manual edits to a scalar string tensor.')
    print('Syntax: edit TENSOR')
    print('Note: the operation is performed in-memory. To write changes back to the checkpoint, run `commit` after `edit`.')

  def complete_edit(self, text, line, begidx, endidx):
    return self.complete_cat(text, line, begidx, endidx)

  def do_edit(self, arg):
    arg = arg.split()
    if len(arg) != 1:
      print('edit: invalid usage.')
      return
    target = self._cwd.find(arg[0])
    if target is None:
      print('{}: not found.'.format(arg[0]))
    elif not target.is_terminal:
      print('{}: not a tensor.'.format(arg[0]))
    else:
      name = target.full_name
      if target.full_name in self._loads:
        tensor = self._loads[name]
        # If numpy byte string, convert to a decoded Python string.
        if tensor.dtype.kind == 'S':
          tensor = tensor.tostring().decode()
      else:
        # If the tensor was renamed but not committed, find the original name so we can look it up
        # in the checkpoint file.
        tensor = tf.train.load_variable(self._checkpoint, self._original_name(name))

      if not isinstance(tensor, bytes):
        print('{}: not a string tensor.'.format(arg[0]))
        return

      with tempfile.NamedTemporaryFile(suffix='.tmp') as fp:
        fp.write(tensor)
        fp.flush()
        ret = subprocess.call([EDITOR, fp.name])
        if ret == 0:
          fp.seek(0)
          new_tensor = fp.read()
          if new_tensor != tensor:
            self._loads[name] = np.array(new_tensor)
        else:
          print('edit - not storing modifications; editor exited with non-zero status.')

  def help_mv(self):
    print('mv - move/rename tensor or directory.')
    print('Syntax: mv SRC DEST')
    print('Note: the operation is performed in-memory. To write changes back to the checkpoint, run `commit` after `mv`.')

  def _add_or_update(self, old_name, new_name):
    for k, v in self._renames.items():
      if v == old_name:
        self._renames[k] = new_name
        return
    self._renames[old_name] = new_name

  def do_mv(self, arg):
    arg = arg.split()
    if len(arg) != 2:
      print('mv: invalid usage.')
      return
    src, dest = arg
    src = self._cwd.find(src)
    if src is None:
      print('{}: invalid source.'.format(arg[0]))
      return
    mutations = self._cwd.move(src, dest)
    if mutations is not None:
      # Rename pending loads if needed.
      for old_name, new_name in mutations:
        self._add_or_update(old_name, new_name)
        if old_name in self._loads:
          self._loads[new_name] = self._loads.pop(old_name)
    else:
      print('mv: cannot relink {} to {}'.format(arg[0], arg[1]))

  def help_mutations(self):
    print('mutations - list all in-memory move/rename/load operations that have not been written to disk yet.')
    print('Syntax: mutations')

  def do_mutations(self, arg):
    for src, dest in sorted(self._renames.items()):
      print('[Rename] {} -> {}'.format(src, dest))
    for key, _ in sorted(self._loads.items()):
      print('[Load] {}'.format(key))

  def help_commit(self):
    print('commit - writes all pending mutations to the checkpoint.')
    print('Syntax: commit')

  def do_commit(self, arg):
    '''Commits all pending changes to the checkpoint.'''

    all_vars = self._all_vars()
    needs_repair = False

    # TODO: add documentation for repair argument.
    if '--repair' in arg.split():
      needs_repair = True in ['//' in name for name in all_vars]

    if not self._dirty and not needs_repair:
      print('Nothing to commit.')
      return

    def commit(replacements, loads):
      tf.reset_default_graph()
      with tf.Session() as session:
        for name in all_vars:
          var = tf.train.load_variable(self._checkpoint, name)
          if name in replacements:
            name = replacements[name]
          if name in loads:
            var = loads[name]
            loads.pop(name)
          if needs_repair and '//' in name:
            name = re.sub(r'/+', r'/', name)
          var = tf.Variable(var, name=name)
        # Add new variables to checkpoint if they didn't exist before.
        for name, value in loads.items():
          var = tf.Variable(value, name=name)
        session.run(tf.global_variables_initializer())
        tf.train.Saver().save(session, self._checkpoint, write_meta_graph=False, write_state=False)

    commit(self._renames, self._loads)
    self._renames = {}
    self._loads = {}

  def help_exit(self):
    print('exit - exits the shell.')
    print('Syntax: exit')

  def do_exit(self, arg):
    return self.do_EOF(1)

  def help_EOF(self):
    print('^D - exits the shell.')
    print('Syntax: ^D')

  def do_EOF(self, arg):
    if not arg:
      print('exit')
    if self._dirty:
      print('WARNING: there are pending mutations that have not been written to disk. Discard (y/N)? ', end='', flush=True)
      line = sys.stdin.readline().strip().lower()
      if line == 'y' or line == 'yes':
        print('')
        return True
      else:
        print('You can view the pending mutations with the `mutations` command or write them out to disk with `commit`.')
        return
    print()
    return True

  def _original_name(self, name):
    for src, dest in self._renames.items():
      if name == dest:
        return src
    return name

  def _all_vars(self):
    return [var_name for var_name, _ in tf.train.list_variables(self._checkpoint)]

  def _build_tree(self, names):
    root = Node(None, '', is_terminal=False)
    for name in names:
      root.insert(name, is_terminal=True)
    return root

  @property
  def _dirty(self):
    return len(self._renames) > 0 or len(self._loads) > 0


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('Usage: {} CHECKPOINT'.format(sys.argv[0]))
    sys.exit(-1)

  # Ugh.
  import tensorflow as tf
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
  tf.logging.set_verbosity(tf.logging.ERROR)
  ExplorerShell(sys.argv[1]).cmdloop()
