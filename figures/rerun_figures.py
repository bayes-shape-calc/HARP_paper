ms_scripts = [
	"figures/rayleigh_overlap.py",
	# "figures/global_adfscan.py", ## long time, >20 min
	"figures/EPres_year.py",
	"figures/EPres_resolution_model.py",
	"figures/EPres_imgs.py",
	"figures/EPres_imgs_highres.py",
	"figures/EPres_imgs_lowres.py",
	"figures/EPres_radial.py",
	"figures/EPres_residue_aa.py"
]


si_scripts = [
	"figures/theoretical_size.py",
	# "figures/global_blobsigma.py", ## ~6 min
	"figures/inter_residue_distances_allpdb.py",
	"figures/EPres_camera.py",
	"figures/EPres_reconstructionsoftware.py",
	"figures/EPres_month.py",
	"figures/EPres_journal.py",
	"figures/EPres_dose.py",
	"figures/EPres_voltage.py",
	"figures/EPres_humidity.py",
	"figures/EPres_FW.py",
	"figures/EPres_residue_rg.py",
	"figures/decarboxylation/decarb.py",
	"figures/EPres_residue_others.py",

]
for sn in ms_scripts:
	exec(open("%s"%(sn)).read())

for sn in si_scripts:
	exec(open("%s"%(sn)).read())