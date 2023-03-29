#ifndef __LOGGING_
#define __LOGGING_

#include <cstdio>
#include <cstring>
#include "color.hpp"
#include "constants.hpp"

#define RULELEN     92
#define PREFIX      "c "

#if defined(__linux__) || defined(__CYGWIN__)
#pragma GCC system_header
#endif

#define PUTCH(CH, ...) putc(CH, stdout);

#define PRINT(FORMAT, ...) fprintf(stdout, FORMAT, ## __VA_ARGS__);

inline void REPCH(const char& ch, const size_t& size, const size_t& off = 0) {
    for (size_t i = off; i < size; i++) PUTCH(ch);
}

#define QLOGS(RESULT) \
    do { \
        if (!quiet_en) QLOG0(""); \
        PRINT("s %s\n", RESULT); \
    } while (0)

#define QLOGE(FORMAT, ...) \
  do { \
     SETCOLOR(CERROR, stderr); \
     fprintf(stderr, "ERROR - "); \
     fprintf(stderr, FORMAT, ## __VA_ARGS__); \
     putc('\n', stderr); \
     SETCOLOR(CNORMAL, stderr); \
     exit(1); \
  } while (0)

#define QLOGEN(FORMAT, ...) \
  do { \
     SETCOLOR(CERROR, stderr); \
     fprintf(stderr, "ERROR - "); \
     fprintf(stderr, FORMAT, ## __VA_ARGS__); \
     putc('\n', stderr); \
     SETCOLOR(CNORMAL, stderr); \
  } while (0)

#define QLOGW(FORMAT, ...) \
  do { \
     SETCOLOR(CWARNING, stderr); \
     fprintf(stderr, "WARNING - ");\
     fprintf(stderr, FORMAT, ## __VA_ARGS__);\
     putc('\n', stderr); \
     SETCOLOR(CNORMAL, stderr); \
  } while (0)

#define QLRULER(CH, TIMES) \
  do { \
     PUTCH(PREFIX[0]); \
     REPCH(CH, TIMES);      \
     PUTCH('\n'); \
  } while (0)

#define QNAME(NAME, VER) \
  QLRULER('-', RULELEN); \
  do { \
     const char* suffix = "SAT Solver (version "; \
     size_t len = strlen(NAME) + strlen(suffix) + strlen(VER) + 1; \
     if (RULELEN < len) QLOGE("ruler length is smaller than the title"); \
     size_t gap = (RULELEN - len - 3) / 2; \
     PRINT(PREFIX); PUTCH(' '); \
     REPCH(' ', gap); \
     PRINT("%s%s%s %s%s%s)%s", UNDERLINE, CSOLVER, NAME, CSOLVER, suffix, VER, CNORMAL); \
     REPCH(' ', RULELEN + 1, len + gap + 3), PUTCH('\n'); \
  } while (0); \

#define QAUTHORS(AUTHORS) \
  do { \
     const char *prefix = "Authors: "; \
     const char *suffix = ", all rights reserved"; \
     size_t len = strlen(prefix) + strlen(AUTHORS) + strlen(suffix); \
     if (RULELEN < len) QLOGE("ruler length is smaller than the authors"); \
     size_t gap = RULELEN - len - 1; \
     PRINT(PREFIX); PUTCH(' '); \
     PRINT("%s%s%s%s%s", prefix, CAUTHOR, AUTHORS, suffix, CNORMAL); \
     REPCH(' ', gap), PUTCH('\n'); \
  } while (0); \

#define QLOG0(MESSAGE) do { PRINT(PREFIX); PRINT("%s\n", MESSAGE); } while (0)

#define QLOGN0(MESSAGE) do { PRINT(PREFIX); PRINT("%s", MESSAGE); } while (0)

#define QLOG1(FORMAT, ...) \
    do { \
        PRINT(PREFIX); PRINT(FORMAT, ## __VA_ARGS__); PUTCH('\n'); \
    } while (0)

#define QLOGN1(FORMAT, ...) \
    do { \
        PRINT(PREFIX); PRINT(FORMAT, ## __VA_ARGS__); \
    } while (0)

#define QLOG2(VERBOSITY, FORMAT, ...) \
    do { \
        if (verbose >= VERBOSITY) { PRINT(PREFIX); PRINT(FORMAT, ## __VA_ARGS__); PUTCH('\n'); } \
    } while (0)

#define QLOGN2(VERBOSITY, FORMAT, ...) \
    do { \
        if (verbose >= VERBOSITY) { PRINT(PREFIX); PRINT(FORMAT, ## __VA_ARGS__); } \
    } while(0)

#define QPRINT(VERBOSITY, MAXVERBOSITY, FORMAT, ...) \
    do { \
        if (verbose >= VERBOSITY && verbose < MAXVERBOSITY) { PRINT(FORMAT, ## __VA_ARGS__); } \
    } while (0)

#define QLDONE(VERBOSITY, MAXVERBOSITY) if (verbose >= VERBOSITY && verbose < MAXVERBOSITY) PRINT("done.\n");

#define QLENDING(VERBOSITY, MAXVERBOSITY, FORMAT, ...) \
    do { \
        if (verbose >= VERBOSITY && verbose < MAXVERBOSITY) { \
            PRINT(FORMAT, ## __VA_ARGS__); \
            PRINT(" done.\n"); \
        } \
    } while(0)

#define QLMEMCALL(SOLVER, VERBOSITY) QLOG2(VERBOSITY, " Memory used in %s call: %lld MB", __func__, sysMemUsed() / MBYTE);

#define QLGCMEM(VERBOSITY, oldB, newB) \
    if (verbose >= VERBOSITY) { \
        double diff = abs(double(oldB.size() - newB.size())) * oldB.bucket(); \
        PRINT("(%.2f KB saved) ", diff / KBYTE); \
    }

#define QLREDALL(SOLVER, VERBOSITY, MESSAGE) \
    if (verbose >= VERBOSITY) { \
        SOLVER->countAll(); \
        SOLVER->updateNumElims(); \
        QLOG1("\t\t %s%s%s", CLBLUE, MESSAGE, CNORMAL); \
        SOLVER->logReductions(); }

#define QLREDALLHOST(SOLVER, VERBOSITY, MESSAGE) \
    if (verbose >= VERBOSITY) { \
        SOLVER->countAll(1); \
        SOLVER->updateNumElims(); \
        QLOG1("\t\t %s", MESSAGE); \
        SOLVER->logReductions(); }

#define QLREDCL(SOLVER, VERBOSITY, MESSAGE) \
    if (verbose >= VERBOSITY) { \
        SOLVER->countAll(); \
        inf.n_del_vars_after = 0; \
        QLOG1("\t\t %s%s%s", CLBLUE, MESSAGE, CNORMAL); \
        SOLVER->logReductions(); }

#define QORGINF(SOLVER, CLS, LITS) \
    int64 CLS = SOLVER->stats.clauses.original; \
    int64 LITS = SOLVER->stats.literals.original; \

#define QLEARNTINF(SOLVER, CLS, LITS) \
    int64 CLS = SOLVER->stats.clauses.learnt; \
    int64 LITS = SOLVER->stats.literals.learnt; \

#define QLSHRINKALL(SOLVER, VERBOSITY, BCLS, BLITS) \
    do { \
        int64 RCLS = BCLS - maxClauses(), RLITS = BLITS - maxLiterals(); \
        SOLVER->stats.shrink.clauses += RCLS, SOLVER->stats.shrink.literals += RLITS; \
        QLENDING(VERBOSITY, 5, "(-%lld clauses, -%lld literals)", RCLS, RLITS); \
    } while (0)

#define QLSHRINKORG(SOLVER, VERBOSITY, BCLS, BLITS) \
    do { \
        int64 RCLS = BCLS - SOLVER->stats.clauses.original, RLITS = BLITS - SOLVER->stats.literals.original; \
        SOLVER->stats.shrink.clauses += RCLS, SOLVER->stats.shrink.literals += RLITS; \
        QLENDING(VERBOSITY, 5, "(-%lld clauses, -%lld literals)", RCLS, RLITS); \
    } while (0)

#define QLSHRINKLEARNT(SOLVER, VERBOSITY, BCLS, BLITS) \
    do { \
        int64 RCLS = BCLS - SOLVER->stats.clauses.learnt, RLITS = BLITS - SOLVER->stats.literals.learnt; \
        SOLVER->stats.shrink.clauses += RCLS, SOLVER->stats.shrink.literals += RLITS; \
        QLENDING(VERBOSITY, 5, "(-%lld clauses, -%lld literals)", RCLS, RLITS); \
    } while (0)

#ifdef LOGGING

#define QLBCPS(SOLVER, VERBOSITY, LIT) \
     if (verbose >= VERBOSITY) { \
		QLOG1("\t Before BCP(%d)", l2i(LIT)); \
		SOLVER->printWatched(ABS(LIT)); }

#define QLBCP(SOLVER, VERBOSITY, LIT) \
     if (verbose >= VERBOSITY) { \
		QLOG1("\t BCP(%d)", l2i(LIT)); \
		SOLVER->printOL(LIT); \
        SOLVER->printOL(FLIP(LIT)); }

#define QLTRAIL(SOLVER, VERBOSITY) if (verbose >= VERBOSITY) SOLVER->printTrail();

#define QLLEARNT(SOLVER, VERBOSITY) if (verbose >= VERBOSITY) SOLVER->printLearnt();

#define QLSORTED(SOLVER, SIZE, VERBOSITY) if (verbose >= VERBOSITY) SOLVER->printSortedStack(SIZE);

#define QLCLAUSE(VERBOSITY, CLAUSE, FORMAT, ...) \
    do { \
        if (verbose >= VERBOSITY) { \
            PRINT(PREFIX);\
            PRINT(FORMAT, ## __VA_ARGS__); \
            SETCOLOR(CLOGGING, stdout);\
            CLAUSE.print(); \
            SETCOLOR(CNORMAL, stdout);\
        } \
    } while (0)

#define QLDL(SOLVER, VERBOSITY) QLOG2(VERBOSITY, " Current decision level: %d", SOLVER->DL());

#define QLBCPE(SOLVER, VERBOSITY, LIT) \
     if (verbose >= VERBOSITY) { \
		QLOG1("\t After BCP(%d)", l2i(LIT)); \
		SOLVER->printWatched(ABS(LIT)); \
        QLRULER('-', 30); }

#define QLOCCURS(SOLVER, VERBOSITY, VAR) \
     if (verbose >= VERBOSITY) { \
		QLOG1("\t Full Occurrence LIST(%d)", VAR); \
		SOLVER->printOccurs(VAR); \
        QLRULER('-', 30); }

#define QLNEWLIT(SOLVER, VERBOSITY, SRC, LIT) \
    do { \
        QLOG2(VERBOSITY, "   %sNew %s( %d@%d )%s", CREPORTVAL, SRC == NOREF ? !SOLVER->DL() ? "forced unit" : "decision" : "unit", l2i(LIT), l2dl(LIT), CNORMAL); \
    } while (0)

#define QLCONFLICT(SOLVER, VERBOSITY, LIT) \
    do { \
        QLOG2(VERBOSITY, " %sConflict detected in literal( %d@%d )%s", CCONFLICT, l2i(LIT), l2dl(LIT), CNORMAL); \
    } while (0)

#else // NO LOGGING

#define QLBCPS(SOLVER, VERBOSITY, LIT) do { } while (0)

#define QLBCP(SOLVER, VERBOSITY, LIT) do { } while (0)

#define QLTRAIL(SOLVER, VERBOSITY) do { } while (0)

#define QLLEARNT(SOLVER, VERBOSITY) do { } while (0)

#define QLSORTED(SOLVER, SIZE, VERBOSITY) do { } while (0)

#define QLCLAUSE(VERBOSITY, CLAUSE, FORMAT, ...) do { } while (0)

#define QLDL(SOLVER, VERBOSITY) do { } while (0)

#define QLBCPE(SOLVER, VERBOSITY, LIT) do { } while (0)

#define QLOCCURS(SOLVER, VERBOSITY, VAR) do { } while (0)

#define QLNEWLIT(SOLVER, VERBOSITY, SRC, LIT) do { } while (0)

#define QLCONFLICT(SOLVER, VERBOSITY, LIT) do { } while (0)

#endif // NO LOGGING


#endif
