"""Embedd Mayavi in Eelbrain

for testing:

src = datasets.get_mne_sample(src='ico', sub=[0])['src']
brain = plot.brain.brain(src.source, mask=False,hemi='lh',views='lat')
"""
from traits.trait_base import ETSConfig
ETSConfig.toolkit = 'wx'

from mayavi.core.ui.api import SceneEditor, MlabSceneModel
from traits.api import HasTraits, Instance
from traitsui.api import View, Item, HGroup, VGroup
from tvtk.api import tvtk
import wx

from .._wxgui.frame import EelbrainFrame


SCENE_NAME = 'scene_%i'


class MayaviView(HasTraits):

    view = Instance(View)

    def __init__(self, width, height, n_rows, n_columns):
        HasTraits.__init__(self)

        n_scenes = n_rows * n_columns
        if n_scenes < 1:
            raise ValueError("n_rows=%r, n_columns=%r" % (n_rows, n_columns))

        self.scenes = tuple(MlabSceneModel() for _ in xrange(n_scenes))
        for i, scene in enumerate(self.scenes):
            self.add_trait(SCENE_NAME % i, scene)

        if n_rows == n_columns == 1:
            self.view = View(Item(SCENE_NAME % 0, editor=SceneEditor(),
                                  resizable=True, show_label=False),
                             width=width, height=height, resizable=True)
        else:
            rows = []
            for row in xrange(n_rows):
                columns = []
                for column in xrange(n_columns):
                    i = row * n_columns + column
                    item = Item(SCENE_NAME % i, editor=SceneEditor(),
                                resizable=True, show_label=False)
                    columns.append(item)
                rows.append(HGroup(*columns))
            self.view = View(VGroup(*rows))

        self.figures = [scene.mayavi_scene for scene in self.scenes]


class BrainFrame(EelbrainFrame):

    def __init__(self, parent, title, width, height, n_rows, n_columns):
        EelbrainFrame.__init__(self, parent, wx.ID_ANY, title)
        self.mayavi_view = MayaviView(width, height, n_rows, n_columns)
        # Use traits to create a panel, and use it as the content of this
        # wx frame.
        self.ui = self.mayavi_view.edit_traits(parent=self,
                                               view=self.mayavi_view.view,
                                               kind='subpanel')
        self.panel = self.ui.control
        # Hide the toolbar (the edit_traits command assigns scene_editor)
        for scene in self.mayavi_view.scenes:
            scene.scene_editor._tool_bar.Show(False)
            scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

        self.panel.SetSize((width, height))
        self.Fit()

        self.figure = self.mayavi_view.figures
