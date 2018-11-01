#!/usr/bin/env python

import cmd
import getopt
import numpy as np
import os
import sys


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
    self._mutations = []

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
    for name, shape in tf.contrib.framework.list_variables(self._checkpoint):
      print('{} : [{}]'.format(name, 'x'.join(map(str, shape))))

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
      for k, v in reversed(self._mutations):
        if v == name:
          name = k
      print(tf.contrib.framework.load_variable(self._checkpoint, name))

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
      name = target.full_name
      for k, v in reversed(self._mutations):
        if v == name:
          name = k
      tensor = tf.contrib.framework.load_variable(self._checkpoint, name)
      try:
        np.save(arg[1], tensor, allow_pickle=False)
      except Exception as e:
        print(str(e))

  def help_mv(self):
    print('mv - move/rename tensor or directory.')
    print('Syntax: mv SRC DEST')
    print('Note: the operation is performed in-memory. To write changes back to the checkpoint, run `commit` after `mv`.')

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
      self._mutations += mutations
    else:
      print('mv: cannot relink {} to {}'.format(arg[0], arg[1]))

  def help_mutations(self):
    print('mutations - list all in-memory move/rename operations that have not been written to disk yet.')
    print('Syntax: mutations')

  def do_mutations(self, arg):
    for src, dest in self._mutations:
      print('{} -> {}'.format(src, dest))

  def help_commit(self):
    print('commit - writes all pending mutations to the checkpoint.')
    print('Syntax: commit')

  def do_commit(self, arg):
    '''Commits all pending changes to the checkpoint.'''
    if len(self._mutations) == 0:
      print('Nothing to commit.')
      return

    def commit(replacements):
      tf.reset_default_graph()
      session = tf.Session()
      for name in self._all_vars():
        var = tf.contrib.framework.load_variable(self._checkpoint, name)
        if name in replacements:
          print('Replaced {} with {}'.format(name, replacements[name]))
          name = replacements[name]
        var = tf.Variable(var, name=name)
      saver = tf.train.Saver()
      session.run(tf.global_variables_initializer())
      saver.save(session, self._checkpoint)

    seen = set()
    replacements = {}
    while len(self._mutations) > 0:
      k, v = self._mutations.pop(0)
      if k in seen:
        commit(replacements)
        seen = set([v])
        replacements = {k: v}
      else:
        replacements[k] = v
        seen.add(v)

    if len(replacements) > 0:
      commit(replacements)

  def help_exit(self):
    print('exit - exits the shell.')
    print('Syntax: exit')

  def do_exit(self, arg):
    self.do_EOF(self, arg)

  def help_EOF(self):
    print('^D - exits the shell.')
    print('Syntax: ^D')

  def do_EOF(self, arg):
    print('exit')
    if len(self._mutations) > 0:
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

  def _all_vars(self):
    return [var_name for var_name, _ in tf.contrib.framework.list_variables(self._checkpoint)]

  def _build_tree(self, names):
    root = Node(None, '', is_terminal=False)
    for name in names:
      root.insert(name, is_terminal=True)
    return root


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('Usage: {} CHECKPOINT'.format(sys.argv[0]))
    sys.exit(-1)

  # Ugh.
  import tensorflow as tf
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
  tf.logging.set_verbosity(tf.logging.ERROR)
  ExplorerShell(sys.argv[1]).cmdloop()
