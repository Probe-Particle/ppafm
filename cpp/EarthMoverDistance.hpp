#ifndef EarthMoverDistance_h
#define EarthMoverDistance_h

/// see https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221Zn48MeBHK1f1riH4l1acuGTY6p7KlYK9%22%5D,%22action%22:%22open%22,%22userId%22:%22100958146796876347936%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing


#include <cstdlib> // For qsort, new, delete (though new/delete are C++ keywords)
#include <cmath>   // For fabs
#include <cstdio>  // For printf (for error messages and potentially debugging)

// --- Forward declaration for the C interface if needed later ---
class Wasserstein1D;
typedef Wasserstein1D* Wasserstein1D_Handle;


// Structure to hold value-weight pairs for sorting
struct VWPair { // Value-Weight Pair
    double v;   // value
    double w;   // weight
};

// Comparison function for qsort (sorts VWPair by value)
int compare_vw_pairs_qsort(const void* a, const void* b) {
    VWPair* pair_a = (VWPair*)a;
    VWPair* pair_b = (VWPair*)b;
    if (pair_a->v < pair_b->v) return -1;
    if (pair_a->v > pair_b->v) return 1;
    return 0;
}

// Comparison function for qsort (sorts doubles)
int compare_doubles_qsort(const void* a, const void* b) {
    double val_a = *(const double*)a;
    double val_b = *(const double*)b;
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    return 0;
}


class Wasserstein1D { public:

    size_t max_n1_;
    size_t max_n2_;
    size_t max_n_sum_; // max_n1_ + max_n2_

    VWPair* vw1;  // Sorted value-weight pairs for dist1
    VWPair* vw2;  // Sorted value-weight pairs for dist2
    double* uval; // Unique sorted values from both
    double* cdf1; // CDF of dist1 at uval points
    double* cdf2; // CDF of dist2 at uval points

    bool successfully_allocated_;

    double wTrashold = 1e-9;


    Wasserstein1D(size_t max_n1, size_t max_n2) :
        max_n1_(max_n1), max_n2_(max_n2), max_n_sum_(max_n1 + max_n2),
        vw1(nullptr), vw2(nullptr), uval(nullptr), cdf1(nullptr), cdf2(nullptr),
        successfully_allocated_(false)
    {
        if (max_n_sum_ < max_n1_ || max_n_sum_ < max_n2_) { // Basic overflow check
            fprintf(stderr, "Wasserstein1D Constructor: Max N1 + Max N2 exceeds size_t capacity.\n");
            return;
        }
        if (max_n1_ > 0) { vw1 = new VWPair[max_n1_];   }
        if (max_n2_ > 0) { vw2 = new VWPair[max_n2_];   }
        if (max_n_sum_ > 0) {
            uval = new double[max_n_sum_];
            cdf1 = new double[max_n_sum_];
            cdf2 = new double[max_n_sum_];
        }
        successfully_allocated_ = true;
    }

    ~Wasserstein1D() {
        cleanup();
    }

    // Delete copy constructor and assignment operator to prevent shallow copies
    Wasserstein1D(const Wasserstein1D&) = delete;
    Wasserstein1D& operator=(const Wasserstein1D&) = delete;
    
    bool is_valid() const {   return successfully_allocated_; }


    void cleanup() {
        //printf("Wasserstein1D::cleanup() \n");
        delete[] vw1; vw1     = nullptr;
        delete[] vw2; vw2     = nullptr;
        delete[] uval; uval   = nullptr;
        delete[] cdf1; cdf1   = nullptr;
        delete[] cdf2; cdf2   = nullptr;
        successfully_allocated_ = false; 
    }

    double sort_vw_pairs(VWPair* vw, const double* v_in, const double* w_in, size_t n) {
        // --- Normalize weights and prepare VWPair arrays ---
        if (n > 0) {
            double sum_w = 0.0;
            for (size_t i = 0; i < n; ++i) {
                vw[i].v = v_in[i];
                vw[i].w = (w_in) ? w_in[i] : 1.0;
                sum_w += vw[i].w;
            }
            if (sum_w <= wTrashold) { return sum_w; }
            for (size_t i = 0; i < n; ++i) vw[i].w /= sum_w;
            qsort(vw, n, sizeof(VWPair), compare_vw_pairs_qsort);
        }
        return sum_w;
    }

    double calculate_cdf(VWPair* vw, size_t n, double* uval, size_t nuval) {
        double cdf = 0.0;
        size_t i = 0;
        for (size_t k = 0; k < nuval; ++k) {
            double val_k = uval[k];
            // Accumulate weights for values <= val_k
            while (i < n && vw[i].v <= val_k + wTrashold) {
                cdf += vw[i].w;
                i++;
            }
        }
        return cdf;
    }


    double calculate(
        const double* v1_in, const double* w1_in, size_t n1,
        const double* v2_in, const double* w2_in, size_t n2
    ) {
        if (!successfully_allocated_) { fprintf(stderr, "Wasserstein1D::calculate: Object not successfully initialized (memory allocation failed).\n"); return -1.0; }
        if (n1 > max_n1_ || n2 > max_n2_) { fprintf(stderr, "Wasserstein1D::calculate: Error: Input size n1=%zu or n2=%zu exceeds pre-allocated max_n1=%zu or max_n2=%zu.\n",n1, n2, max_n1_, max_n2_); return -1.0; }
        if ((n1 == 0 && n2 == 0) || (n1 > 0 && v1_in == nullptr) || (n2 > 0 && v2_in == nullptr)) { if (n1 == 0 && n2 == 0) return 0.0; fprintf(stderr, "Wasserstein1D::calculate: Error: Null input value array for non-zero size.\n"); return -1.0; }

        double sum_w1 = sort_vw_pairs(vw1, v1_in, w1_in, n1);
        if (sum_w1 <= wTrashold) {  fprintf(stderr, "Wasserstein1D::calculate: Error: Sum of weights for distribution 1 is non-positive.\n");  return -1.0;}

        double sum_w2 = sort_vw_pairs(vw2, v2_in, w2_in, n2);
        if (sum_w2 <= wTrashold) {   fprintf(stderr, "Wasserstein1D::calculate: Error: Sum of weights for distribution 2 is non-positive.\n");  return -1.0;}

        if (n1 == 0 || n2 == 0) { // One is empty
            return 0.0; // Policy: distance to/from empty distribution is 0
        }

        // --- Create sorted list of unique values (uval_) ---
        // First, copy all values into uval_, then sort and unique-ify
        size_t nuval_ = 0;
        for(size_t k=0; k<n1; ++k) uval[nuval_++] = vw1[k].v;
        for(size_t k=0; k<n2; ++k) uval[nuval_++] = vw2[k].v;
        if (nuval_ > 0) {  qsort(uval, nuval_, sizeof(double), compare_doubles_qsort);}

        size_t nuval = 0; // Actual count of unique values
        if (nuval_ > 0) {
            uval[0] = uval[0]; // Keep the first element
            nuval = 1;
            for (size_t k = 1; k < nuval_; ++k) {
                // Compare with a small epsilon for floating point uniqueness
                if (uval[k] > uval[nuval - 1] + wTrashold) { uval[nuval++] = uval[k]; }
            }
        }

        if (nuval == 0) return 0.0; 

        // --- Calculate CDFs at unique points ---
        double cdf1 = calculate_cdf(vw1, n1, uval, nuval);
        double cdf2 = calculate_cdf(vw2, n2, uval, nuval);

        // --- Calculate distance ---
        double dist = 0.0;
        if (nuval > 1) { // Need at least two unique points to form an interval
            for (size_t k = 0; k < nuval - 1; ++k) {
                double cdf_d = fabs(cdf1[k] - cdf2[k]);
                double interval_w = uval[k+1] - uval[k];
                // interval_w should be > 0 due to unique sort with epsilon,
                // but a check for robustness doesn't hurt if extremely close values exist.
                if (interval_w > 1e-12) { // Use a smaller epsilon for interval width check
                    dist += cdf_d * interval_w;
                }
            }
        }
        return dist;
    }
};

// --- Extern "C" Interface ---
extern "C" {
    Wasserstein1D_Handle Wasserstein1D_Create(size_t max_n1, size_t max_n2) {
        Wasserstein1D* calc = new (std::nothrow) Wasserstein1D(max_n1, max_n2);
        if (calc && !calc->is_valid()) { // Allocation succeeded but internal setup failed
            delete calc;
            calc = nullptr;
            fprintf(stderr, "Wasserstein1D_Create: Internal allocation failed within constructor.\n");
        } else if (!calc) {
             fprintf(stderr, "Wasserstein1D_Create: Memory allocation failed for Wasserstein1D object.\n");
        }
        return calc;
    }

    void Wasserstein1D_Destroy(Wasserstein1D_Handle handle) {
        if (handle) {
            delete handle;
        }
    }

    double Wasserstein1D_Calculate_C( // Renamed to avoid conflict if class is in global scope
        Wasserstein1D_Handle handle,
        const double* v1, const double* w1, size_t n1,
        const double* v2, const double* w2, size_t n2
    ) {
        if (!handle) {
            fprintf(stderr, "Wasserstein1D_Calculate_C: Invalid handle.\n");
            return -1.0; // Indicate error
        }
        // No try-catch needed here as the C++ class methods don't throw exceptions
        // but rather print to stderr and return error codes.
        return handle->calculate(v1, w1, n1, v2, w2, n2);
    }
}


#endif // SVD_H
