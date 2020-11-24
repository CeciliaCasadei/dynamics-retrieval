from pymol import cmd
import sys
import getopt


def myfunc_bcolor():
    cmd.load('./6g7h_edited_nonH.pdb')
    #cmd.color('blue', selection=' (name C*)')  
    cmd.hide('all')
    cmd.cartoon('tube', 'all')
    cmd.set('cartoon_transparency', 0.3, 'all')
    cmd.show('cartoon')    
    cmd.bg_color('white')
    #cmd.hide('all')
    cmd.select('sel', '((chain A and (resi 57 or resi 82 or resi 85 or resi 182 or resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))')
    cmd.show('lines', 'sel')    
    cmd.show('spheres', 'sel')
    cmd.set('sphere_scale', 0.10, 'all')
    cmd.spectrum('b', 'white_magenta')
    cmd.spectrum('b', 'white_magenta', 'sel')
    cmd.ray(2048, 1024)
    cmd.png('./test_bcolor.png')
        



def myfunc_bins(): 
    cmd.load('./6g7h.pdb')
    cmd.show('sticks', 'all')
    cmd.bg_color('white')
    cmd.hide('all')
    cmd.select('sel', '((chain A and (resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))')
    cmd.show('lines', 'sel')    
    cmd.show('spheres', 'sel')
    cmd.set('sphere_scale', 0.10, 'all')
    f = open('list.txt', 'r')
    for i in f:
	print i.split()
	fn = i.split()[8]
    
    #for i in range(18):
        #bin_left = str(i)
        #bin_right = str(i+2)
        
        #cmd.load('./map_bins/modes_1_3_bin_%s_%s.ccp4'%(bin_left, bin_right), 'nlsa_map')
	cmd.load('./map_bins/%s'%(fn), 'nlsa_map')
        cmd.zoom('sel')
        cmd.set_view('\
                     0.306473434,    0.183651671,   -0.933994949,\
                     -0.874885261,   0.440932453,   -0.200377792,\
                     0.375029773,    0.878551126,    0.295809299,\
                     0.000000000,    0.000000000,   -52.589275360,\
                     17.967073441,   40.217617035,   33.910476685,\
                     40.623615265,   64.554939270,  -20.000000000' )
        
        cmd.isomesh('map_modes_1_2_p4', 'nlsa_map', 4.0, 'sel', carve=2.0)
        cmd.color('cyan', 'map_modes_1_2_p4')
        cmd.isomesh('map_modes_1_2_m4', 'nlsa_map', -4.0, 'sel', carve=2.0)
        cmd.color('purple', 'map_modes_1_2_m4')
        cmd.set('mesh_width', 0.5)
        cmd.set('fog_start', 0.1)
        cmd.show('mesh', 'map_modes_1_2_p4')
        cmd.show('mesh', 'map_modes_1_2_m4')
        cmd.ray(2048, 1024)
        #cmd.png('bin_%s_%s_nlsa_modes_1_3.png'%(bin_left, bin_right))
        cmd.png('./FRAMES_bin_wat/%s.png'%fn)


        cmd.delete('map_modes_1_2_p4')
        cmd.delete('map_modes_1_2_m4')
        cmd.delete('nlsa_map')
        
        #cmd.load('../binning_Science/bin_%s_%s.ccp4'%(bin_left, bin_right), 'bin_map')
        #cmd.zoom('sel')
        #cmd.set_view('\
        #             0.306473434,     0.183651671,   -0.933994949,\
        #             -0.874885261,    0.440932453,   -0.200377792,\
        #             0.375029773,     0.878551126,    0.295809299,\
        #             0.000000000,     0.000000000,   -52.589275360,\
        #             17.967073441,    40.217617035,   33.910476685,\
        #             40.623615265,    64.554939270,  -20.000000000' )
        
        #cmd.isomesh('map_bin_p4', 'bin_map', 4.0, 'sel', carve=2.0)
        #cmd.color('cyan', 'map_bin_p4')
        #cmd.isomesh('map_bin_m4', 'bin_map', -4.0, 'sel', carve=2.0)
        #cmd.color('purple', 'map_bin_m4')
        #cmd.set('mesh_width', 0.5)
        #cmd.set('fog_start', 0.1)
        #cmd.show('mesh', 'map_bin_p4')
        #cmd.show('mesh', 'map_bin_m4')
        #cmd.ray(2048, 1024)
        #cmd.png('bin_%s_%s.png'%(bin_left, bin_right))
        
        #cmd.delete('map_bin_p4')
        #cmd.delete('map_bin_m4')
        #cmd.delete('bin_map')

def myfunc_step(myArguments):  
    try:
        optionPairs, leftOver = getopt.getopt(myArguments, "h", ["inputMode="])
    except getopt.GetoptError:
        print 'Usage: ...'
        sys.exit(2)   
    for option, value in optionPairs:
        if option == '-h':
            print 'Usage: ...'
            sys.exit()
        elif option == "--inputMode":
            mode = int(value)
  
    print 'MODE: ', mode
    cmd.load('./6g7h.pdb')
    cmd.color('blue', selection=' (name C*)')
    cmd.load('./6g7k.pdb')
    cmd.show('sticks', 'all')
    cmd.bg_color('white')
    cmd.hide('all')
    cmd.select('sel', '((chain A and (resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))')
    cmd.show('lines', 'sel')    
    cmd.show('spheres', 'sel')    
    cmd.set('sphere_scale', 0.10, 'all')
    
    for time in range(0, 114600, 100):
		        
        cmd.load('./1.5_bR_light_mode_0_%d_timestep_%0.6d_light--dark_bR_dark_mode_0_avg.ccp4'%(mode, time), 'nlsa_map')
		
        cmd.set_view ('\
		               0.306473434,    0.183651671,   -0.933994949,\
		              -0.874885261,    0.440932453,   -0.200377792,\
		               0.375029773,    0.878551126,    0.295809299,\
		               0.200000018,    2.000000000,  -43.038898468,\
		              17.967073441,   40.217617035,   33.910476685,\
		             -45.510078430,  131.587860107,  -20.000000000 ')        
        cmd.isomesh('map_modes_0_x_p4', 'nlsa_map', 4.0, 'sel', carve=2.0)
        cmd.color('cyan', 'map_modes_0_x_p4')
        cmd.isomesh('map_modes_0_x_m4', 'nlsa_map', -4.0, 'sel', carve=2.0)
        cmd.color('purple', 'map_modes_0_x_m4')
        cmd.set('mesh_width', 0.3)
        cmd.set('fog_start', 0.1)
        cmd.show('mesh', 'map_modes_0_x_p4')
        cmd.show('mesh', 'map_modes_0_x_m4')
        cmd.ray(2048, 1024)
        cmd.png('./FRAMES_light_mode_0_%d_4sig/nlsa_light_modes_0_%d_time_%d_4sigma.png'%(mode, mode, time))
		
        cmd.delete('map_modes_0_x_p4')
        cmd.delete('map_modes_0_x_m4')
        cmd.delete('nlsa_map')
        
def myfunc_C20():    
    cmd.load('./6g7h.pdb')
    cmd.color('blue', selection=' (name C*)')
    cmd.load('./6g7k.pdb')
    cmd.show('sticks', 'all')
    cmd.bg_color('white')
    cmd.hide('all')
    cmd.select('sel', '((chain A and (resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))')
    cmd.show('lines', 'sel')    
    cmd.show('spheres', 'sel')    
    cmd.set('sphere_scale', 0.10, 'all')
    
    for time in range(100, 99200, 100):
        cmd.load('./map_mode_0_2_phase_6g7h/1.5_bR_light_mode_0_2_timestep_%d_light--dark_bR_dark_mode_0_avg.ccp4'%(time), 'nlsa_map')
        cmd.zoom('sel')
        cmd.set_view ('\
                      19696312,    0.989641845,   -0.079237513,\
                      955249608,   0.136545390,    0.262389272,\
                      70491034,    0.044284314,    0.961704612,\
                      799999952,   0.000000000,   -20.975885391,\
                      967073441,   40.217617035,   33.910476685,\
                      .573089600,  109.524818420,  -20.000000000' )
        cmd.isomesh('map_modes_0_x_p4', 'nlsa_map', 3.5, 'sel', carve=2.0)
        cmd.color('cyan', 'map_modes_0_x_p4')
        cmd.isomesh('map_modes_0_x_m4', 'nlsa_map', -3.5, 'sel', carve=2.0)
        cmd.color('purple', 'map_modes_0_x_m4')
        cmd.set('mesh_width', 0.3)
        cmd.set('fog_start', 0.1)
        cmd.show('mesh', 'map_modes_0_x_p4')
        cmd.show('mesh', 'map_modes_0_x_m4')
        cmd.ray(2048, 1024)
        cmd.png('./FRAMES_mode_0_2_phase_6g7h_3p5sig_C20_b/nlsa_modes_0_2_time_%d.png'%(time))
        
        cmd.delete('map_modes_0_x_p4')
        cmd.delete('map_modes_0_x_m4')
        cmd.delete('nlsa_map')

def myfunc_penta():
    mode = 5  
    cmd.load('./6g7h.pdb')
    cmd.color('lime', selection=' (name C*)')
    cmd.show('sticks', 'all')
    cmd.bg_color('white')
    cmd.hide('all')
    cmd.select('sel', '((chain A and (resi 57 or resi 82 or resi 85 resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))')
    cmd.show('lines', 'sel')    
    cmd.show('spheres', 'sel')
    
    cmd.set('sphere_scale', 0.10, 'all')
    
    for time in range(100, 99200, 100):
        cmd.load('./map_mode_0_%d_phase_6g7h/1.5_bR_light_mode_0_%d_timestep_%d_light--dark_bR_dark_mode_0_avg.ccp4'%(mode, mode, time), 'nlsa_map')
        cmd.zoom('sel')
        cmd.set_view('\
     			   0.894949734,   -0.088602446,    0.437280059,\
     			   0.444157988,    0.269824058,   -0.854351640,\
    			       -0.042292103,    0.958824337,    0.280833066,\
     			   0.200000018,    2.000000000,  -43.038898468,\
    			       17.967073441,   40.217617035,   33.910476685,\
   			       -40.510078430,  126.587860107,  -20.000000000 ')
        
        cmd.isomesh('map_modes_0_x_p4', 'nlsa_map', 4.0, 'sel', carve=2.0)
        cmd.color('cyan', 'map_modes_0_x_p4')
        cmd.isomesh('map_modes_0_x_m4', 'nlsa_map', -4.0, 'sel', carve=2.0)
        cmd.color('purple', 'map_modes_0_x_m4')
        cmd.set('mesh_width', 0.3)
        cmd.set('fog_start', 0.1)
        cmd.show('mesh', 'map_modes_0_x_p4')
        cmd.show('mesh', 'map_modes_0_x_m4')
        cmd.clip('slab', 20)
        cmd.clip('far', +6)
        cmd.ray(2048, 1024)
        cmd.png('./FRAMES_mode_0_%d_phase_6g7h_4sig_penta/nlsa_modes_0_%d_time_%d.png'%(mode, mode, time))
        
        cmd.delete('map_modes_0_x_p4')
        cmd.delete('map_modes_0_x_m4')
        cmd.delete('nlsa_map')

from pymol.cgo import *

def myfunc_sphere():
    
    cmd.load('./6g7h.pdb')
    cmd.color('blue', selection=' (name C*)')
    #cmd.load('./6g7k.pdb')
    cmd.show('sticks', 'all')
    cmd.bg_color('white')
    cmd.hide('all')
    cmd.select('sel', '((chain A and (resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))')
    cmd.show('lines', 'sel')    
    cmd.show('spheres', 'sel')
    cmd.set_view ('\
     			0.306473434,    0.183651671,   -0.933994949,\
    			    -0.874885261,    0.440932453,   -0.200377792,\
     			0.375029773,    0.878551126,    0.295809299,\
    			    -2.000000000,    0.000000000,  -58.213687897,\
    			    16.646999359,   42.976001740,   33.564998627,\
    			    55.213687897,   61.213691711,  -20.000000000 ')
    
    cmd.set('sphere_scale', 0.10, 'all')
    
#    spherelist = [
#   			    COLOR,    0.100,    0.900,    0.000,
#   			    ALPHA, 0.5,
#   			    #SPHERE,   16.647,  42.976,  33.565, 1.70, # C10
#                 SPHERE, 18.083,  40.207,  36.736, 1.70, # C20
#    		        ]	
#
#    cmd.load_cgo(spherelist, 'segment',   1)
    cmd.ray(2048, 1024)
    #cmd.png('./6g7h_sphere_1p7A_C20.png')
    cmd.png('./6g7h.png')




def myfunc_cartoon():
    modes = [2]
    #modes = [1, 2, 3, 4, 5]
    cmd.load('./6g7h.pdb')
    #cmd.color('blue', selection=' (name C*)')  
    cmd.hide('all')
    cmd.cartoon('tube', 'all')
    cmd.set('cartoon_transparency', 0.7, 'all')
    cmd.set('cartoon_side_chain_helper', 'on')
    cmd.show('cartoon') 
    cmd.spectrum('resi', 'rainbow', 'all')
    cmd.bg_color('white')
    #cmd.hide('all')
    cmd.select('sel', '((chain A and (resi 47-57 or resi 82 or resi 85 or resi 182 or resi 206 or resi 208 or resi 212 or resi 216 or resn RET) and not h.) or (chain A and (resi 407 or resi 411 or resi 415)))')
    cmd.show('lines', 'sel')    
    cmd.show('spheres', 'sel')
    cmd.set_view('\
    0.711140990,    0.037647203,   -0.702040076,\
    -0.699935138,   -0.055949945,   -0.712014735,\
    -0.066083595,    0.997723699,   -0.013440915,\
     0.000000000,    1.000000000,  -62.475360870,\
    17.146160126,   39.064826965,   35.888786316,\
    17.158218384,  107.792404175,  -20.000000000  ')
    #cmd.set_view ('\
    #              0.396753609,    0.062696703,   -0.915781319,\
    #              -0.904319465,   -0.144434780,   -0.401676387,\
    #              -0.157453746,    0.987527311,   -0.000608850,\
    #              0.000000000,    0.000000000, -177.199661255,\
    #              17.146160126,   39.064826965,   35.888786316,\
    #              131.882507324,  222.516723633,  -20.000000000' )
    
    cmd.set('sphere_scale', 0.08, 'all')
    
    for mode in modes:
	    print mode
	    for time in range(38200, 99200, 100):#(100, 99200, 100)
		cmd.load('./map_mode_0_%d_phase_6g7h/1.5_bR_light_mode_0_%d_timestep_%d_light--dark_bR_dark_mode_0_avg.ccp4'%(mode, mode, time), 'nlsa_map')        	        
		#cmd.load('./MAPS_dark_mode_0_2/1.5_bR_dark_mode_0_2_timestep_%d_light--dark_bR_dark_mode_0_avg.ccp4'%(time), 'nlsa_map')
		#cmd.zoom('sel')
		cmd.set_view('\
	    0.711140990,    0.037647203,   -0.702040076,\
	    -0.699935138,   -0.055949945,   -0.712014735,\
	    -0.066083595,    0.997723699,   -0.013440915,\
	     0.000000000,    1.000000000,  -62.475360870,\
	    17.146160126,   39.064826965,   35.888786316,\
	    17.158218384,  107.792404175,  -20.000000000  ')
		
		cmd.isomesh('map_modes_0_x_p4', 'nlsa_map', 4.5, 'all', carve=5.0)
		cmd.color('cyan', 'map_modes_0_x_p4')
		cmd.isomesh('map_modes_0_x_m4', 'nlsa_map', -4.5, 'all', carve=5.0)
		cmd.color('purple', 'map_modes_0_x_m4')
		cmd.set('mesh_width', 0.3)
		cmd.set('fog_start', 0.1)
		cmd.show('mesh', 'map_modes_0_x_p4')
		cmd.show('mesh', 'map_modes_0_x_m4')
		#cmd.clip('slab', 20)
		#cmd.clip('far', +6)
		cmd.ray(2048, 1024)
		#cmd.png('TEST.png')
		cmd.png('./FRAMES_light_mode_0_%d_4p5sig/cartoon_nlsa_modes_0_%d_time_%d_4p5sigma.png'%(mode, mode, time))
		
		cmd.delete('map_modes_0_x_p4')
		cmd.delete('map_modes_0_x_m4')
		cmd.delete('nlsa_map')
	    

cmd.extend('myfunc_bins',    myfunc_bins)
cmd.extend('myfunc_step',    myfunc_step)
cmd.extend('myfunc_C20',     myfunc_C20)
cmd.extend('myfunc_penta',   myfunc_penta)
cmd.extend('myfunc_sphere',  myfunc_sphere)
cmd.extend('myfunc_cartoon', myfunc_cartoon)
cmd.extend('myfunc_bcolor',  myfunc_bcolor)


print "\n**** CALLING myfunc_step ****"
myfunc_step(sys.argv[1:])  


