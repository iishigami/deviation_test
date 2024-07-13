import cProfile
import pstats
import io
from plotter import Plotter

def profile_plotter():
    data_file = 'deviation.json'
    plotter = Plotter(data_file)
    plotter.draw_plots()

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    profile_plotter()
    pr.disable()
    
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    
    with open("profile_output.txt", "w") as f:
        f.write(s.getvalue())
