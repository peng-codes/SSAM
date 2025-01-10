#ifndef __StopWatch_H
#define __StopWatch_H

#include <windows.h>
#include <vector>
#include <exception>

class StopWatchWin 
{
    public:
        //! Constructor, default
        StopWatchWin() :
            start_time(),     end_time(),
            diff_time(0.0f),  total_time(0.0f),
            running(false), clock_sessions(0), freq(0), freq_set(false)
        {
            if (! freq_set)
            {
                // helper variable
                LARGE_INTEGER temp;

                // get the tick frequency from the OS
                QueryPerformanceFrequency((LARGE_INTEGER *) &temp);

                // convert to type in which it is needed
                freq = ((double) temp.QuadPart) / 1000.0;

                // rememeber query
                freq_set = true;
            }
        };

        // Destructor
        ~StopWatchWin() { };

    public:
        //! Start time measurement
        inline void start();

        //! Stop time measurement
        inline void stop();

        //! Reset time counters to zero
        inline void reset();

        //! Time in msec. after start. If the stop watch is still running (i.e. there
        //! was no call to stop()) then the elapsed time is returned, otherwise the
        //! time between the last start() and stop call is returned
        inline float getTime();

        //! Mean time to date based on the number of times the stopwatch has been
        //! _stopped_ (ie finished sessions) and the current total time
        inline float getAverageTime();

    private:
        // member variables

        //! Start of measurement
        LARGE_INTEGER  start_time;
        //! End of measurement
        LARGE_INTEGER  end_time;

        //! Time difference between the last start and stop
        float  diff_time;

        //! TOTAL time difference between starts and stops
        float  total_time;

        //! flag if the stop watch is running
        bool running;

        //! Number of times clock has been started
        //! and stopped to allow averaging
        int clock_sessions;

        //! tick frequency
        double  freq;

        //! flag if the frequency has been set
        bool  freq_set;
};

// functions, inlined

////////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchWin::start()
{
    QueryPerformanceCounter((LARGE_INTEGER *) &start_time);
    running = true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement and increment add to the current diff_time summation
//! variable. Also increment the number of times this clock has been run.
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchWin::stop()
{
    QueryPerformanceCounter((LARGE_INTEGER *) &end_time);
    diff_time = (float)
                (((double) end_time.QuadPart - (double) start_time.QuadPart) / freq);

    total_time += diff_time;
    clock_sessions++;
    running = false;
}

////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchWin::reset()
{
    diff_time = 0;
    total_time = 0;
    clock_sessions = 0;

    if (running)
    {
        QueryPerformanceCounter((LARGE_INTEGER *) &start_time);
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned added to the
//! current diff_time sum, otherwise the current summed time difference alone
//! is returned.
////////////////////////////////////////////////////////////////////////////////
inline float
StopWatchWin::getTime()
{
    // Return the TOTAL time to date
    float retval = total_time;

    if (running)
    {
        LARGE_INTEGER temp;
        QueryPerformanceCounter((LARGE_INTEGER *) &temp);
        retval += (float)
                  (((double)(temp.QuadPart - start_time.QuadPart)) / freq);
    }

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. for a single run based on the total number of COMPLETED runs
//! and the total time.
////////////////////////////////////////////////////////////////////////////////
inline float
StopWatchWin::getAverageTime()
{
    return (clock_sessions > 0) ? (total_time/clock_sessions) : 0.0f;
}













#endif //__StopWatch_H