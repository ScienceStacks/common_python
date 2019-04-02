'''Extends list.'''

class ExtendedList(list):

  def removeAll(self, element):
    while self.count(element) > 0:
      self.remove(element)
