'''Extends list.'''

class ExtendedList(list):

  def removeAll(self, element):
    while self.count(element) > 0:
      self.remove(element)

  def unique(self):
    """
    Returns a list of unique elements.
    Does not preserve order.
    """
    new_list = ExtendedList(list(set(self)))
    [self.pop() for _ in range(len(self))]
    [self.append(e) for e in new_list]
