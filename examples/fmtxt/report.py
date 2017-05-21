from eelbrain import *

# uncomment the following line to embed plots as PNG instead of the default SVG
# plot.configure(format='png')


# create the initial report
report = Report("Test Report", "Test Author")

# add a new section to the report and add some text
s1 = report.add_section("The Big Analysis")
s1.append("This is an introductory paragraph adding some text. It's important "
          "to have some text. What follows is a table showing the first ten "
          "rows of the data, so that it may be examined thoroughly.")

# Add a table to the section
ds = datasets.get_uv()
t = ds.as_table(cases=10, midrule=True, caption="The first ten rows of the "
                                                "data table.")
s1.append(t)

# add a subsection with more text and another table
s11 = s1.add_section("Averages")
s11.append("This is a subordinate section which will show the average of the "
           "data.")

ds_ = ds.aggregate("A%B", drop=['rm'])
t = ds_.as_table(cases=0, midrule=True, caption="Averages of the data.")
s11.append(t)

# add a second subsection with a figure
s12 = s1.add_section("Figure")
s12.append("And now finally we will show a figure")
p = plot.Boxplot('fltvar', 'A%B', 'rm', ds=ds)
image = report.add_image_figure(p, "Boxplot of all the data")
p.close()

# add a more complex figure with legend
ds = datasets.get_uts()
p = plot.UTSStat('uts', 'A%B', match='rm', ds=ds, legend=False)
p_legend = p.plot_legend()
report.add_figure("Two plots in one figure", [p, p_legend])
p.close()
p_legend.close()

# add information on package versions used
report.sign()

# print string representation and save
print(report.get_str())
report.save_html('test_report')
