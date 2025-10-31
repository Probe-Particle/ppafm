
#ifndef logging_h
#define logging_h

#include <stdio.h>
#include <stdarg.h>

#include <string>

int log_fd = 1;

void log(const char* log_level, const char* format, ...) {
    std::string log_level_str = "LOG_ENTRY_" + std::string(log_level) + std::string(" ");
    std::string format_str = log_level_str + std::string(format);
    va_list arg_list;
    va_start(arg_list, format);
    printf("log_fd %d %p\n", log_fd, &log_fd);
    vdprintf(log_fd, format_str.c_str(), arg_list);
}

void log_info(const char* format, ...) {
    va_list arg_list;
    va_start(arg_list, format);
    log("INFO", format, arg_list);
}

void set_log_fd_(int fd) {
    printf("Setting fd %d %d %p\n", log_fd, fd, &log_fd);
    log_fd = fd;
    printf("Setting fd %d %d %p\n", log_fd, fd, &log_fd);
}

#endif
