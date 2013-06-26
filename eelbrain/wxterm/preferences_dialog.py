'''
Created on Nov 8, 2011

@author: christian
'''
import logging
import os

import wx

from . import ID


class PreferencesDialog(wx.Frame):
    def __init__(self, shell, Id=wx.ID_ANY, pos=wx.DefaultPosition,
                 size=(500, 50), style=wx.DEFAULT_FRAME_STYLE):  # wx.DefaultSize
        """
        tutorial on layout:
        http://zetcode.com/wxpython/layout/

        """
        title = "Eelbrain Preferences"
        wx.Frame.__init__(self, shell, Id, title, pos, size, style)
        self.config = shell.wx_config

        pref_sizer = wx.BoxSizer(wx.VERTICAL)

    # Startup Script ---
        panel = wx.Panel(self, -1)  # , size=(500,300))
        panel.SetBackgroundColour("BLUE")
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        txt = wx.StaticText(panel, label="Startup Script:")
        sizer.Add(txt, 0, wx.ALIGN_LEFT)

        btn = wx.Button(panel, label="Show File")
        btn.Bind(wx.EVT_BUTTON, self.OnShowStartupScript)
        sizer.Add(btn, 0, wx.ALIGN_LEFT)

        btn = wx.Button(panel, label="Edit")
        btn.Bind(wx.EVT_BUTTON, self.OnEditStartupScript)
        sizer.Add(btn, 0, wx.ALIGN_LEFT)

        btn = wx.Button(panel, wx.ID_HELP)
        btn.Bind(wx.EVT_BUTTON, self.OnHelpStartupScript)
        sizer.Add(btn, 0, wx.ALIGN_RIGHT)

        pref_sizer.Add(sizer, 0)

    # Font ---
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        button = wx.Button(panel, ID.SET_FONT, "Font")
        self.Bind(wx.EVT_BUTTON, self.OnSetFont, id=ID.SET_FONT)
        sizer.Add(button, 0, wx.ALIGN_LEFT | wx.EXPAND)
        pref_sizer.Add(sizer, 0)

        panel.SetSizer(pref_sizer)

        pref_sizer.Fit(self)

    def OnHelpStartupScript(self, event):
        msg = ("The startup script is executed every time the Eelbrain "
               "application is opened. If the PYTHONSTARTUP environment "
               "variable is defined, the file designated by the path stored "
               "in this variable is used instead of Eelbrain's startup "
               "script.")
        dlg = wx.MessageDialog(self, msg, "Help: Startup Script",
                               wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def OnEditStartupScript(self, event):
        path = self.Parent.startupScript
        if not os.path.exists(path):
            with open(path, 'w') as fid:
                fid.write("# Eelbrain startup script\n")
        self.Parent.create_py_editor(pyfile=path)

    def OnSetFont(self, event):
        try:
            cur_size = int(self.config.Read('font size'))
        except:
            cur_size = 12

        font = wx.Font(pointSize=cur_size,
                       family=wx.FONTFAMILY_UNKNOWN,
                       style=wx.FONTSTYLE_NORMAL,
                       weight=wx.FONTWEIGHT_NORMAL,
                       face=self.config.Read('font'))
        data = wx.FontData()
        data.EnableEffects(True)
        data.SetInitialFont(font)
        data.SetColour(self.config.Read('font color'))
        dlg = wx.FontDialog(self, data)
        if dlg.ShowModal() == wx.ID_OK:
            data = dlg.GetFontData()
            font_ = data.GetChosenFont()
            font = font_.GetFaceName()
            size = font_.GetPointSize()
            color_ = data.GetColour()
            color = color_.GetAsString(wx.C2S_HTML_SYNTAX)

            logging.debug('You selected: "%s", %d points, color %s\n' %
                          (font, size, color))

            self.config.Write("font", font)
            self.config.Write("font size", str(size))
            self.config.Write("font color", color)
            self.Parent.ApplyStyle()

        dlg.Destroy()

    def OnShowStartupScript(self, event):
        path = os.path.dirname(self.Parent.startupScript)
        os.system('open %s' % path)



