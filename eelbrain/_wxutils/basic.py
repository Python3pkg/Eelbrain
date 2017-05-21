'''Some WxPython utilities'''
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import re

import wx
from wx.lib.dialogs import ScrolledMessageDialog

from . import icons


# store icons once loaded for repeated access
_cache = {}
_iconcache = {}


def Icon(path, asicon=False):
    if asicon:
        if path not in _iconcache:
            _iconcache[path] = icons.catalog[path].GetIcon()
        return _iconcache[path]
    else:
        if path not in _cache:
            _cache[path] = icons.catalog[path].GetBitmap()
        return _cache[path]


def show_text_dialog(parent, text, caption):
    "Create and show a ScrolledMessageDialog"
    style = wx.CAPTION | wx.CLOSE_BOX | wx.RESIZE_BORDER | wx.SYSTEM_MENU
    dlg = ScrolledMessageDialog(parent, text, caption, style=style)
    font = wx.Font(12, wx.MODERN, wx.NORMAL, wx.NORMAL, False, 'Inconsolata')
    dlg.text.SetFont(font)

    n_lines = dlg.text.GetNumberOfLines()
    line_text = dlg.text.GetLineText(0)
    w, h = dlg.text.GetTextExtent(line_text)
    dlg.text.SetSize((w + 100, (h + 3) * n_lines + 50))

    dlg.Fit()
    dlg.Show()
    return dlg


class REValidator(wx.PyValidator):
    "Ensure that the value of a text field matches a regular expression"
    def __init__(self, pattern, message, can_be_empty=False):
        wx.PyValidator.__init__(self)
        self.pattern = re.compile(pattern)
        self.message = message
        self.can_be_empty = bool(can_be_empty)

    def Clone(self):
        return REValidator(self.pattern, self.message, self.can_be_empty)

    def Validate(self, win):
        ctrl = self.GetWindow()
        text = ctrl.GetValue()

        if len(text.strip()) == 0 and self.can_be_empty:
            return True

        if self.pattern.match(text):
            return True

        wx.MessageBox(self.message.format(value=text), "Error")
        ctrl.SetBackgroundColour("pink")
        ctrl.SetFocus()
        ctrl.Refresh()
        return False
#         else:
#             ctrl.SetBackgroundColour(
#                 wx.SystemSettings_GetColour(wx.SYS_COLOUR_WINDOW))
#             ctrl.Refresh()
#             return True

    def TransferToWindow(self):
        return True

    def TransferFromWindow(self):
        return True
