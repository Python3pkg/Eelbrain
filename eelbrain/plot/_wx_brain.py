"""Embedd Mayavi in Eelbrain

for testing:

src = datasets.get_mne_sample(src='ico', sub=[0])['src']
brain = plot.brain.brain(src.source, mask=False,hemi='lh',views='lat')
"""
from traits.trait_base import ETSConfig
ETSConfig.toolkit = 'wx'

from mayavi.core.ui.api import SceneEditor, MlabSceneModel
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
import wx

from .._wxgui.frame import EelbrainFrame


class MayaviView(HasTraits):

    scene = Instance(MlabSceneModel, ())

    # The layout of the panel created by Traits
    view = View(Item('scene',
                     editor=SceneEditor(),
                     resizable=True,
                     show_label=False),
                resizable=True)

    def __init__(self):
        HasTraits.__init__(self)


class BrainFrame(EelbrainFrame):

    def __init__(self, parent, title, w, h):
        EelbrainFrame.__init__(self, parent, wx.ID_ANY, title)
        self.mayavi_view = MayaviView()
        # Use traits to create a panel, and use it as the content of this
        # wx frame.
        self.panel = self.mayavi_view.edit_traits(
            parent=self,
            kind='subpanel').control

        self.panel.SetSize((w, h))
        self.Fit()

        self.figure = [self.mayavi_view.scene.mayavi_scene]
