
#include "logging.h"

extern "C"{

    void set_log_fd(int fd) {
        set_log_fd_(fd);
    }

    void test_log() {
        log_info("test message 1");
        // log_debug("test message 2\nanother line");
    }

}
