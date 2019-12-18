/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  //creating the normal Gaussian distributions
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  std::default_random_engine gen;

  //initializing all particles
  for (int i=0; i<num_particles; i++) {
    particles[i].id = i;
    particles[i].weight = 1.0;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);

    //initializing all weights
    weights[i] = 1.0;
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

   //creating the normal Gaussian distributions
   normal_distribution<double> dist_x(0, std_pos[0]);
   normal_distribution<double> dist_y(0, std_pos[1]);
   normal_distribution<double> dist_theta(0, std_pos[2]);

   std::default_random_engine gen;

   for (int i=0; i<num_particles; i++) {
     // car is moving straight
     if (fabs(yaw_rate) < 0.00001) {
       particles[i].x += velocity * delta_t * cos(particles[i].theta);
       particles[i].y += velocity * delta_t * sin(particles[i].theta);
     }
     //car is not moving straight
     else {
       particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
       particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
       particles[i].theta += yaw_rate * delta_t;
     }
   }

   //adding noise!
   particles[i].x += dist_x(gen);
   particles[i].y += dist_y(gen);
   particles[i].theta += dist_theta(gen);
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */

   for (int i=0; i<observations.size(); i++) {
     //initializing the minimum distance to a very large number
     double min_dist = numeric_limits<double>::max();

     for (int j=0; j<predicted.size(); j++) {
       double current_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

       if (min_dist > current_dist) {
         min_dist = current_dist;
         observations[i].id = predicted[j].id;
       }
     }
   }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

   //calculating values needed for updating weights using multivariate Gaussian
   double gaussian = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
   double gauss_x = 2 * pow(std_landmark[0], 2);
   double gauss_y = 2 * pow(std_landmark[1], 2);

   for (int i=0; i<num_particles; i++) {
     //resetting the weights
     particles[i].weight = 1.0;

     vector<LandmarkObs> close_landmarks;

     for (int j=0; j<map_landmarks.landmark_list.size(); j++) {
       double current_dist = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);

       //check if the landmark is whiting the sensor's range
       if (current_dist < sensor_range) {
         close_landmarks.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
       }
     }

     vector<LandmarkObs> transformed_obs;

     for (int j=0; j<observations.size(); j++) {
       transformed_x = (observations[j].x * cos(particles[i].theta)) - (observations[j].y * sin(particles[i].theta)) + particles[i].x;
       transformed_y = (observations[j].x * sin(particles[i].theta)) + (observations[j].y * cos(particles[i].theta)) + particles[i].y;

       transformed_obs.push_back(LandmarkObs{observations[j].id, transformed_x, transformed_y});
     }

     //associating valid landmarks to transformed observations
     dataAssociation(close_landmarks, transformed_obs);

     for (int j=0; j<transformed_obs.size(); j++) {
       particles[i].weight *= gaussian * exp(-(pow(transformed_obs[j].x - close_landmarks[transformed_obs[j].id].x, 2) / gauss_x
                                            + pow(transformed_obs[j].y - close_landmarks[transformed_obs[j].id].y, 2) / gauss_y));
     }
     weights[i] = particles[i].weight;
   }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

   //implementing Sebastian Thrun's Resampling Wheel method
   std::random_device rd;
   std::mt19937 gen(rd());
   std::discrete_distribution<> dist_index(0, num_particles-1);
   std::discrete_distribution<> dist_weights(weights.begin(), wights.end());
   std::default_random_engine gen;
   double max_weight = *max_element(weights.begin(), weights.end());
   double beta = 0.0;
   int index = dist_index(gen);

   std::vector<Particle> resampled_particles;

   for (int i=0; i<num_particles; i++) {
     beta += dist_weights * 2.0 * max_weight;

     while (beta > weights[index]) {
       beta -= weights[index];
       index = (index + 1) % num_particles;
     }
     resampled_particles.push_back(particles[index]);
   }

   particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
